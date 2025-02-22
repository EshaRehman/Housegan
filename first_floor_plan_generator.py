import random

class FirstFloorPlanGenerator:
    """
    We define 3 approaches for the 1st-floor layout:
      Approach #1: Garage -> Storage; optionally rename one Bedroom -> Study; no balconies.
      Approach #2: Garage -> Study; exactly one Bedroom has a balcony if it touches boundary.
      Approach #3: Garage -> Study; that Study also gets a balcony if boundary;
                   all Bedrooms get balconies if they touch boundary.

    The method generate_first_floor_plan(approach=None) does:
      - If approach is None => pick randomly from [1,2,3].
      - If approach is 1, 2, or 3 => use exactly that approach.
    """

    def __init__(self, base_floorplan, floor_width, floor_height):
        """
        :param base_floorplan: dict with {room_name: {x, y, width, height}}
        :param floor_width: total width of the floor
        :param floor_height: total height of the floor
        """
        self.base_floorplan = base_floorplan
        self.floor_width = floor_width
        self.floor_height = floor_height

    def generate_first_floor_plan(self, approach=None):
        """
        If approach is None => pick randomly from [1, 2, 3].
        Otherwise use approach 1, 2, or 3 exactly.
        Returns a dict representing the new first-floor layout.
        """
        if approach is None:
            approach = random.choice([1,2,3])

        if approach == 1:
            return self._approach1()
        elif approach == 2:
            return self._approach2()
        else:
            return self._approach3()

    # -----------------------------------------------------
    # Approach #1
    # -----------------------------------------------------
    def _approach1(self):
        """
        1) Garage -> Storage
        2) If >=2 Bedrooms, pick exactly one and rename it -> Study
        3) No balconies
        """
        plan = {}
        # 1) rename Garage -> Storage
        for room_name, rect in self.base_floorplan.items():
            new_rect = dict(rect)
            if "Garage" in room_name:
                new_name = room_name.replace("Garage", "Storage")
                plan[new_name] = new_rect
            else:
                plan[room_name] = new_rect

        # 2) if >=2 bedrooms => rename exactly one => Study
        bedroom_names = [rn for rn in plan if "Bedroom" in rn]
        if len(bedroom_names) >= 2:
            chosen_bed = random.choice(bedroom_names)
            new_key = self._rename_key(plan, chosen_bed, "Study")
            plan[new_key] = plan.pop(chosen_bed)

        # 3) no balconies => done
        return plan

    # -----------------------------------------------------
    # Approach #2
    # -----------------------------------------------------
    def _approach2(self):
        """
        1) Garage -> Study
        2) Exactly one Bedroom gets a balcony if it touches outer boundary
        """
        plan = {}
        # rename Garage -> Study
        for room_name, rect in self.base_floorplan.items():
            new_rect = dict(rect)
            if "Garage" in room_name:
                new_name = room_name.replace("Garage", "Study")
                plan[new_name] = new_rect
            else:
                plan[room_name] = new_rect

        # pick exactly one bedroom => carve out a balcony on boundary
        bedrooms = [rn for rn in plan if "Bedroom" in rn]
        if bedrooms:
            chosen_bedroom = random.choice(bedrooms)
            self._carve_balcony_if_on_boundary(plan, chosen_bedroom)

        return plan

    # -----------------------------------------------------
    # Approach #3
    # -----------------------------------------------------
    def _approach3(self):
        """
        1) Garage -> Study (and if that study touches boundary, carve a balcony).
        2) ALL bedrooms get balconies if they touch boundary.
        """
        plan = {}
        # rename garage->study
        for room_name, rect in self.base_floorplan.items():
            new_rect = dict(rect)
            if "Garage" in room_name:
                new_name = room_name.replace("Garage", "Study")
                plan[new_name] = new_rect
            else:
                plan[room_name] = new_rect

        # the new "Study" might get a balcony if it touches boundary
        study_names = [rn for rn in plan if "Study" in rn]
        for sn in study_names:
            self._carve_balcony_if_on_boundary(plan, sn)

        # all bedrooms => carve balcony if boundary
        bedrooms = [rn for rn in plan if "Bedroom" in rn]
        for b in bedrooms:
            self._carve_balcony_if_on_boundary(plan, b)

        return plan

    # -----------------------------------------------------
    # rename dict key: e.g. "Bedroom_3" => "Study_3"
    # -----------------------------------------------------
    def _rename_key(self, plan, old_key, new_base_name):
        """
        old_key might be "Bedroom_3" => "Study_3"
        If there's no underscore, we just do "Study".
        """
        parts = old_key.split("_", 1)
        if len(parts) == 2:
            return new_base_name + "_" + parts[1]
        else:
            return new_base_name

    # -----------------------------------------------------
    # carve out a balcony if the room touches outer boundary
    # pick exactly one boundary side if multiple
    # -----------------------------------------------------
    def _carve_balcony_if_on_boundary(self, plan, room_name, thickness=1):
        """
        If the specified room touches any outer boundary (left, right, top, bottom),
        carve out a thin balcony on one chosen boundary side.
        """
        if room_name not in plan:
            return

        rect = plan[room_name]
        x, y = rect["x"], rect["y"]
        w, h = rect["width"], rect["height"]

        boundary_sides = []
        # left boundary
        if x == 0:
            boundary_sides.append("left")
        # right boundary
        if x + w == self.floor_width:
            boundary_sides.append("right")
        # bottom boundary
        if y == 0:
            boundary_sides.append("bottom")
        # top boundary
        if y + h == self.floor_height:
            boundary_sides.append("top")

        if not boundary_sides:
            return  # no outer boundary => skip

        side = random.choice(boundary_sides)

        # new balcony room name
        new_balc_name = room_name.replace("Bedroom", "Balcony").replace("Study", "Balcony")
        if new_balc_name == room_name:
            new_balc_name = "Balcony_" + room_name

        # carve out the balcony
        if side == "left":
            if w > thickness:
                balc_rect = {
                    "x": x, "y": y, "width": thickness, "height": h
                }
                rect["x"] += thickness
                rect["width"] -= thickness
                plan[new_balc_name] = balc_rect

        elif side == "right":
            if w > thickness:
                balc_rect = {
                    "x": x + w - thickness,
                    "y": y,
                    "width": thickness,
                    "height": h
                }
                rect["width"] -= thickness
                plan[new_balc_name] = balc_rect

        elif side == "bottom":
            if h > thickness:
                balc_rect = {
                    "x": x, "y": y,
                    "width": w, "height": thickness
                }
                rect["y"] += thickness
                rect["height"] -= thickness
                plan[new_balc_name] = balc_rect

        elif side == "top":
            if h > thickness:
                balc_rect = {
                    "x": x,
                    "y": y + h - thickness,
                    "width": w,
                    "height": thickness
                }
                rect["height"] -= thickness
                plan[new_balc_name] = balc_rect
