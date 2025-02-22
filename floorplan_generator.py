import random

class FloorplanGenerator:
    ROOMS = ["Garage", "Kitchen", "Bedroom", "Washroom"]
    FLOORPLAN_WIDTH = 20
    FLOORPLAN_HEIGHT = 20
    POPULATION_SIZE = 10
    GENERATIONS = 50
    MUTATION_RATE = 0.1

    def __init__(self, rooms=None, attached_washroom=False):
        """
        Initialize the floorplan generator.

        Args:
        - rooms (list): List of rooms to include in the floorplan.
        - attached_washroom (bool): Whether washrooms must be adjacent to bedrooms.
        """
        self.rooms = rooms if rooms else self.ROOMS
        self.attached_washroom = attached_washroom

    @staticmethod
    def check_overlap(room1, room2):
        """
        Check if two rooms overlap.

        Args:
        - room1 (dict): First room's coordinates and dimensions.
        - room2 (dict): Second room's coordinates and dimensions.

        Returns:
        - bool: True if rooms overlap, False otherwise.
        """
        return not (
            room1["x"] + room1["width"] <= room2["x"] or
            room2["x"] + room2["width"] <= room1["x"] or
            room1["y"] + room1["height"] <= room2["y"] or
            room2["y"] + room2["height"] <= room1["y"]
        )

    @staticmethod
    def check_min_gap(room1, room2, min_gap=3):
        """
        Check if there is at least `min_gap` units of space between two rooms.

        Args:
        - room1 (dict): First room's coordinates and dimensions.
        - room2 (dict): Second room's coordinates and dimensions.
        - min_gap (int): Minimum required gap between the rooms.

        Returns:
        - bool: True if the gap is satisfied, False otherwise.
        """
        return (
            room1["x"] + room1["width"] + min_gap <= room2["x"] or
            room2["x"] + room2["width"] + min_gap <= room1["x"] or
            room1["y"] + room1["height"] + min_gap <= room2["y"] or
            room2["y"] + room2["height"] + min_gap <= room1["y"]
        )

    @staticmethod
    def is_flush_adjacent(r1, r2):
        """
        Return True if two rectangles share a full vertical or horizontal boundary.
        That is, they have exactly the same y-range (for horizontal alignment)
        or x-range (for vertical alignment) and are exactly flush.

        For example:
          - rect1.right == rect2.left  AND overlapping y-interval
          - rect1.left == rect2.right  AND overlapping y-interval
          - rect1.top == rect2.bottom  AND overlapping x-interval
          - rect1.bottom == rect2.top  AND overlapping x-interval
        """
        r1_left,  r1_right  = r1["x"],                 r1["x"]+r1["width"]
        r1_bottom,r1_top    = r1["y"],                 r1["y"]+r1["height"]
        r2_left,  r2_right  = r2["x"],                 r2["x"]+r2["width"]
        r2_bottom,r2_top    = r2["y"],                 r2["y"]+r2["height"]

        # Check horizontal flush:
        # e.g., r1_right == r2_left or r1_left == r2_right,
        # with a Y-range overlap that matches the full height (or at least partial).
        # But for them to be considered "fully adjacent," we want at least partial overlap in Y.

        # Right/Left adjacency
        if abs(r1_right - r2_left) < 0.0001:  # flush on the vertical boundary
            # Check vertical overlap
            overlap_height = min(r1_top, r2_top) - max(r1_bottom, r2_bottom)
            if overlap_height > 0:
                return True
        if abs(r2_right - r1_left) < 0.0001:
            overlap_height = min(r1_top, r2_top) - max(r1_bottom, r2_bottom)
            if overlap_height > 0:
                return True

        # Check vertical flush:
        # e.g. r1_top == r2_bottom or r1_bottom == r2_top
        # with an X-range overlap
        if abs(r1_top - r2_bottom) < 0.0001:
            overlap_width = min(r1_right, r2_right) - max(r1_left, r2_left)
            if overlap_width > 0:
                return True
        if abs(r2_top - r1_bottom) < 0.0001:
            overlap_width = min(r1_right, r2_right) - max(r1_left, r2_left)
            if overlap_width > 0:
                return True

        return False

    def initialize_population(self):
        """
        Generate an initial population of floorplans.

        Returns:
        - list: A list of valid floorplans.
        """
        population = []
        while len(population) < self.POPULATION_SIZE:
            floorplan = {}
            valid = True

            for room in self.rooms:
                # Smaller room sizes for demonstration
                if "Bedroom" in room:
                    width, height = random.randint(4, 5), random.randint(4, 5)
                elif "Kitchen" in room:
                    width, height = random.randint(3, 4), random.randint(3, 4)
                elif "Washroom" in room:
                    width, height = random.randint(2, 3), random.randint(2, 3)
                else:
                    # Garage or other
                    width, height = random.randint(4, 6), random.randint(4, 6)

                placed = False
                for _ in range(50):  # Limit retries for efficiency
                    x = random.randint(0, self.FLOORPLAN_WIDTH - width)
                    y = random.randint(0, self.FLOORPLAN_HEIGHT - height)
                    room_rect = {"x": x, "y": y, "width": width, "height": height}

                    # Check no overlap with existing
                    if any(self.check_overlap(room_rect, existing_room)
                           for existing_room in floorplan.values()):
                        continue

                    # If it's a bedroom, ensure it's not flush-adjacent to another bedroom
                    if "Bedroom" in room:
                        # Check adjacency to any existing bedroom
                        bedroom_conflict = False
                        for existing_name, existing_rect in floorplan.items():
                            if "Bedroom" in existing_name:
                                # If flush adjacent => conflict
                                if self.is_flush_adjacent(room_rect, existing_rect):
                                    bedroom_conflict = True
                                    break
                        if bedroom_conflict:
                            continue

                    # Enforce adjacency for washrooms if required
                    if self.attached_washroom and "Washroom" in room:
                        if not self._place_adjacent_to_bedroom(floorplan, room, room_rect):
                            # adjacency failed, try another spot
                            continue

                    # Enforce min gap for non-Bedroom/Washroom combos
                    elif not (
                        "Bedroom" in room or "Washroom" in room
                    ) and any(
                        not self.check_min_gap(room_rect, existing_room, min_gap=3)
                        for existing_room in floorplan.values()
                    ):
                        # gap check failed
                        continue

                    floorplan[room] = room_rect
                    placed = True
                    break

                if not placed:
                    valid = False
                    break

            if valid:
                population.append(floorplan)

        return population

    def _place_adjacent_to_bedroom(self, floorplan, washroom_name, room_rect):
        """
        Attempt to place a washroom flush (adjacent) to a distinct bedroom.
        If a Bedroom has already attached a washroom, skip that bedroom.
        """
        for existing_room_name, existing_rect in floorplan.items():
            if "Bedroom" in existing_room_name:
                if existing_rect.get("has_washroom_attached", False):
                    continue  # already has a washroom

                adjacency_options = [
                    {
                        "x": existing_rect["x"] - room_rect["width"],
                        "y": existing_rect["y"],
                        "width": room_rect["width"],
                        "height": room_rect["height"]
                    },  # Left
                    {
                        "x": existing_rect["x"] + existing_rect["width"],
                        "y": existing_rect["y"],
                        "width": room_rect["width"],
                        "height": room_rect["height"]
                    },  # Right
                    {
                        "x": existing_rect["x"],
                        "y": existing_rect["y"] - room_rect["height"],
                        "width": room_rect["width"],
                        "height": room_rect["height"]
                    },  # Above
                    {
                        "x": existing_rect["x"],
                        "y": existing_rect["y"] + existing_rect["height"],
                        "width": room_rect["width"],
                        "height": room_rect["height"]
                    }   # Below
                ]

                for option in adjacency_options:
                    # Check overlap
                    if any(self.check_overlap(option, existing_r) for existing_r in floorplan.values()):
                        continue

                    # If no overlap => place
                    existing_rect["has_washroom_attached"] = True
                    room_rect.update(option)
                    return True

        return False

    def genetic_algorithm(self):
        """
        Run the genetic algorithm to optimize the floorplan.

        Returns:
        - dict: The best floorplan found.
        """
        population = self.initialize_population()
        for _ in range(self.GENERATIONS):
            # Sort by number of rooms placed (descending)
            population = sorted(population, key=len, reverse=True)
            # Return the top plan
            return population[0] if population else {}
