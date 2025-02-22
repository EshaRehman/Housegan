import os
import cv2
import numpy as np
from math import sqrt

class PrettyFloorplanMaker:
    """
    1) Identify black boundary => floorplan.
    2) Identify other rooms by color => skip adjacency so stairs aren't near them.
    3) Identify living room as white region => find outer boundary.
    4) Place a small pink 'Stairs' box on that living-room boundary that
       does NOT border any other room.
       - First: radial search from boundary points
       - If that fails: find a wall with no color adjacency, place stairs in its center.
    5) We do NOT draw any red outlines—only place the stairs.
    """

    def __init__(self, input_dir="perfect", output_dir="pretty"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Stairs box size
        self.stairs_w = 15
        self.stairs_h = 15

        # Tolerance around each known color
        self.tol = 8

        # Known BGR (blue-green-red) for each "room" we want to avoid adjacency with
        # from your specs:
        # Washroom => R255 G255 B233 => B=233,G=255,R=255
        # Garage   => R197 G227 B237 => B=237,G=227,R=197
        # Kitchen  => R177 G243 B177 => B=177,G=243,R=177
        # Bedroom  => R244 G166 B166 => B=166,G=166,R=244
        self.room_colors = [
            (233, 255, 255),  # washroom
            (237, 227, 197),  # garage
            (177, 243, 177),  # kitchen
            (166, 166, 244)   # bedroom
        ]

        # Minimum contour area for the black boundary
        self.min_floor_area = 2000

    def make_pretty_floorplans(self):
        """
        Main entry: reads images from self.input_dir, places stairs,
        saves them in self.output_dir. No room boundaries are drawn.
        """
        # (Optional) clear old images
        for f in os.listdir(self.output_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                os.remove(os.path.join(self.output_dir, f))

        image_files = [
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for fname in image_files:
            path = os.path.join(self.input_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue

            annotated = self._place_stairs_not_near_rooms(img)
            out_path = os.path.join(self.output_dir, fname)
            cv2.imwrite(out_path, annotated)
            print(f"Saved plan with stairs => {out_path}")

    def _place_stairs_not_near_rooms(self, img):
        """
        1) floorplan_mask = largest black boundary
        2) color_mask = union of known room colors => skip adjacency
        3) living_mask => strictly white inside floorplan_mask
        4) pick largest living contour => place stairs
        """
        annotated = img.copy()
        h, w = annotated.shape[:2]

        # 1) floorplan_mask
        floorplan_mask = self._get_floorplan_mask(annotated)
        if floorplan_mask is None:
            return annotated  # no valid boundary

        # 2) color_mask => for Kitchen, Bedroom, Washroom, Garage
        color_mask = self._build_color_mask(annotated, floorplan_mask)

        # 3) living_mask => strictly white inside floorplan
        living_mask = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                if floorplan_mask[y, x] == 0:
                    continue
                b, g, r = annotated[y, x]
                # define white => b>=240, g>=240, r>=240
                if b>=240 and g>=240 and r>=240:
                    living_mask[y, x] = 255

        # find the largest living contour
        cnts, _ = cv2.findContours(living_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return annotated
        largest_lr = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest_lr) < 10:
            return annotated

        # 4) place stairs => boundary-based search skipping near color
        annotated = self._place_stairs_on_living_boundary(annotated, largest_lr, color_mask)
        return annotated

    def _get_floorplan_mask(self, img):
        """
        Finds the largest black boundary => returns mask of that region (255=inside).
        If none big enough, return None.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # black => invert => findContours
        _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        # pick biggest above min_floor_area
        big_cnts = [c for c in cnts if cv2.contourArea(c) > self.min_floor_area]
        if not big_cnts:
            return None
        largest = max(big_cnts, key=cv2.contourArea)
        mask = np.zeros_like(black_mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        return mask

    def _build_color_mask(self, img, floorplan_mask):
        """
        Build a mask of all known room colors (kitchen, bedroom, washroom, garage)
        inside the floorplan. We'll skip placing stairs near these colors.
        """
        h, w = img.shape[:2]
        total_mask = np.zeros((h, w), dtype=np.uint8)

        for (bC, gC, rC) in self.room_colors:
            lower = np.array([max(0,   bC-self.tol), max(0,   gC-self.tol), max(0,   rC-self.tol)], dtype=np.uint8)
            upper = np.array([min(255,bC+self.tol),  min(255,gC+self.tol),  min(255,rC+self.tol)],  dtype=np.uint8)
            mask = cv2.inRange(img, lower, upper)

            # only keep inside floorplan
            mask = cv2.bitwise_and(mask, floorplan_mask)
            total_mask = cv2.bitwise_or(total_mask, mask)

        return total_mask

    def _place_stairs_on_living_boundary(self, annotated, living_contour, color_mask):
        """
        1) We'll build a mask for living_contour.
        2) We'll gather boundary points => skip those near color => radial search.
        3) If that fails, we find a wall segment not near color => place stairs in center.
        """
        h, w = annotated.shape[:2]
        lr_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(lr_mask, [living_contour], -1, 255, -1)

        boundary_pts = living_contour.reshape(-1, 2)

        # Primary approach => radial search
        placed = self._try_boundary_placement(annotated, boundary_pts, lr_mask, color_mask)
        if placed:
            return annotated

        # Secondary approach => find a continuous boundary segment that is not near color
        placed2 = self._place_on_free_wall_segment(annotated, boundary_pts, lr_mask, color_mask)
        return annotated

    def _try_boundary_placement(self, annotated, boundary_pts, lr_mask, color_mask):
        """
        Normal approach: for each boundary point not near color, do radial search.
        Return True if placed, else False.
        """
        for (x1, y1) in boundary_pts:
            if self._neighbor_color(x1, y1, color_mask):
                continue
            # radial search inward
            for dist in range(5, 40, 5):
                for angle_deg in range(0, 360, 30):
                    rad = np.deg2rad(angle_deg)
                    bx = int(x1 + dist*np.cos(rad))
                    by = int(y1 + dist*np.sin(rad))

                    if self._can_place_stairs_box(bx, by, annotated, lr_mask, color_mask):
                        self._draw_stairs_box(annotated, bx, by)
                        return True
        return False

    def _place_on_free_wall_segment(self, annotated, boundary_pts, lr_mask, color_mask):
        """
        Secondary approach: find the largest continuous segment of the boundary
        that is NOT near any color => place stairs in the "center" of that segment,
        flush against the boundary.

        Steps:
          1) Build a list of boundary points that are not near color => free
          2) Break them into segments based on adjacency
          3) Find the largest segment by perimeter distance
          4) Place stairs in the center of that wall
        """
        # 1) filter boundary points => only those not near color
        free_pts = []
        for (x1,y1) in boundary_pts:
            if not self._neighbor_color(x1, y1, color_mask):
                free_pts.append((x1,y1))
        if len(free_pts)<2:
            return False

        # 2) break into segments
        # We'll consider points consecutive if they're next to each other in boundary array
        # Because the contour is closed, we'll also consider the wrap-around from last->first.
        segments = []
        cur_segment = [free_pts[0]]
        for i in range(1, len(free_pts)):
            prev_pt = free_pts[i-1]
            this_pt = free_pts[i]
            if self._distance(prev_pt, this_pt) < 2.0:
                # continue segment
                cur_segment.append(this_pt)
            else:
                # break
                if len(cur_segment)>1:
                    segments.append(cur_segment)
                cur_segment = [this_pt]
        # wrap-around check
        if len(cur_segment)>1:
            # see if it connects to first point
            if self._distance(cur_segment[-1], free_pts[0]) < 2.0:
                # merge with first segment if it also meets
                if segments:
                    # if there's at least one existing segment, let's see
                    first_segment = segments[0]
                    if self._distance(first_segment[0], cur_segment[-1])<2.0:
                        # merge
                        first_segment.extend(cur_segment)
                else:
                    segments.append(cur_segment)
            else:
                segments.append(cur_segment)

        if not segments:
            return False

        # 3) find the largest segment by perimeter distance
        largest_segment = None
        max_dist = 0.0
        for seg in segments:
            dist = self._segment_length(seg)
            if dist>max_dist:
                max_dist = dist
                largest_segment = seg

        if not largest_segment or len(largest_segment)<2:
            return False

        # 4) place stairs in the center of that wall
        # We'll find the param=0.5 along that segment (i.e. half the distance).
        midx, midy = self._get_segment_midpoint(largest_segment)

        # Now we want to place the stairs "flush" against that boundary, i.e. with a small inward normal.
        # We'll approximate the local direction of the segment near the midpoint,
        # then place the stairs just inside.
        # We find the closest point in largest_segment to (midx,midy) => compute local tangent -> normal

        closest_idx = 0
        closest_d = 999999
        for i, pt in enumerate(largest_segment):
            d = self._distance(pt, (midx,midy))
            if d<closest_d:
                closest_d = d
                closest_idx = i

        # local direction => from segment[closest_idx] to segment[closest_idx+1] if next exists
        # if at the end, use segment[closest_idx-1].
        if closest_idx < len(largest_segment)-1:
            p0 = largest_segment[closest_idx]
            p1 = largest_segment[closest_idx+1]
        else:
            p0 = largest_segment[closest_idx-1]
            p1 = largest_segment[closest_idx]

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        length = sqrt(dx*dx + dy*dy)
        if length<1e-3:
            return False

        # normal (pointing inside) => we guess one side, then we check if it's inside
        # a small offset. We'll just pick (nx, ny) = (-dy, dx).
        nx, ny = -dy, dx
        # test a step inside
        testx = int(midx + 0.5 * nx/length)
        testy = int(midy + 0.5 * ny/length)

        # if that test is outside living => flip normal
        if not self._is_inside_living(testx, testy, lr_mask):
            nx, ny = dy, -dx

        # define a final offset inside => maybe 5 pixels
        offset = 5
        place_x = int(midx + offset*nx/length)
        place_y = int(midy + offset*ny/length)

        # check if we can place stairs there
        if self._can_place_stairs_box(place_x, place_y, annotated, lr_mask, color_mask):
            self._draw_stairs_box(annotated, place_x, place_y)
            return True
        # else fail
        return False

    def _distance(self, p1, p2):
        return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _segment_length(self, seg):
        """
        sum of distances between consecutive points in seg
        """
        length = 0.0
        for i in range(len(seg)-1):
            length += self._distance(seg[i], seg[i+1])
        return length

    def _get_segment_midpoint(self, seg):
        """
        Return the midpoint in param=0.5 sense along the polyline
        """
        total_len = self._segment_length(seg)
        half = total_len*0.5
        dist_so_far = 0.0
        for i in range(len(seg)-1):
            d = self._distance(seg[i], seg[i+1])
            if dist_so_far + d>= half:
                # we find partial
                remain = half - dist_so_far
                ratio = remain/d
                mx = seg[i][0] + ratio*(seg[i+1][0]-seg[i][0])
                my = seg[i][1] + ratio*(seg[i+1][1]-seg[i][1])
                return (mx, my)
            dist_so_far += d
        # if we never found half => return last point
        return seg[-1]

    def _is_inside_living(self, x, y, lr_mask):
        """
        True if (x,y) is inside living_mask
        """
        h, w = lr_mask.shape
        if x<0 or y<0 or x>=w or y>=h:
            return False
        return (lr_mask[y, x]==255)

    def _neighbor_color(self, x, y, color_mask):
        """
        Check 3x3 area around (x,y). If color_mask=255 => near color => skip.
        """
        h, w = color_mask.shape
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                nx = x+dx
                ny = y+dy
                if nx<0 or ny<0 or nx>=w or ny>=h:
                    continue
                if color_mask[ny, nx] == 255:
                    return True
        return False

    def _can_place_stairs_box(self, xA, yA, annotated, lr_mask, color_mask):
        """
        Check if a (stairs_w x stairs_h) box at (xA,yA) is fully inside living area
        and doesn't overlap color_mask (rooms).
        """
        h, w = annotated.shape[:2]
        xB = xA + self.stairs_w
        yB = yA + self.stairs_h
        if xA<0 or yA<0 or xB> w or yB> h:
            return False

        sub_lr = lr_mask[yA:yB, xA:xB]
        if cv2.countNonZero(sub_lr) < self.stairs_w*self.stairs_h:
            return False

        sub_color = color_mask[yA:yB, xA:xB]
        if cv2.countNonZero(sub_color) > 0:
            return False

        return True

    def _draw_stairs_box(self, annotated, xA, yA):
        """
        Draw a small pink box labeled 'Stairs'.
        """
        xB = xA + self.stairs_w
        yB = yA + self.stairs_h
        cv2.rectangle(annotated, (xA, yA), (xB, yB), (200,100,200), -1)
        label_x = xA + (self.stairs_w//2) - 10
        label_y = yA + (self.stairs_h//2)
        cv2.putText(
            annotated, "Stairs",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0,0,0), 1, cv2.LINE_AA
        )
