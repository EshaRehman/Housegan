# app.py

import os
import openai
import json
import shutil

# Import your existing modules (unchanged)
from floorplan_generator import FloorplanGenerator
from floorplan_visualizer import FloorplanVisualizer
from room_type_detector import RoomTypeDetector
from perfect_plan_selector import PerfectPlanSelector
from pretty_floorplan_maker import PrettyFloorplanMaker
from first_floor_plan_generator import FirstFloorPlanGenerator

############################################
# 1) YOUR OPENAI API KEY
############################################
# It's safer to store in environment variable, but for demo:
openai.api_key = "sk-...DuYA"  # <-- replace with your full key

def copy_json_for_png(src_png, src_dir, dst_dir):
    base = os.path.splitext(src_png)[0]
    src_json_path = os.path.join(src_dir, base + ".json")
    if os.path.isfile(src_json_path):
        shutil.copy2(src_json_path, dst_dir)

def rename_png_and_json(old_png, old_dir, new_png, new_dir):
    old_png_path = os.path.join(old_dir, old_png)
    new_png_path = os.path.join(new_dir, new_png)
    if os.path.exists(new_png_path):
        os.remove(new_png_path)
    os.rename(old_png_path, new_png_path)

    old_base = os.path.splitext(old_png)[0]
    new_base = os.path.splitext(new_png)[0]
    old_json = old_base + ".json"
    new_json = new_base + ".json"
    old_json_path = os.path.join(old_dir, old_json)
    new_json_path = os.path.join(new_dir, new_json)
    if os.path.exists(old_json_path):
        if os.path.exists(new_json_path):
            os.remove(new_json_path)
        os.rename(old_json_path, new_json_path)


def parse_floorplan_request(user_text: str) -> dict:
    """
    Use OpenAI to parse user_text for:
      - bedrooms (1..3)
      - washrooms (1..3)
      - has_garage (true/false)
      - has_attachedwashroom (true/false)

    Return a dict like:
      { "bedrooms":2, "washrooms":1, "has_garage":False, "has_attachedwashroom":False }
    """
    system_content = """
    You are a helpful AI that extracts floorplan specs from the user's text.
    They might mention:
      - number of bedrooms (1..3)
      - number of washrooms (1..3)
      - garage or not
      - attached washrooms or not
    If not stated, pick defaults. Return strictly valid JSON with 4 keys:
      bedrooms, washrooms, has_garage, has_attachedwashroom
    Example:
    {
      "bedrooms": 2,
      "washrooms": 2,
      "has_garage": true,
      "has_attachedwashroom": false
    }
    """

    user_prompt = f"User request:\n{user_text}\nExtract the parameters in JSON only."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0
        )
        content = response["choices"][0]["message"]["content"].strip()
        data = json.loads(content)
    except:
        # fallback if parse fails
        data = {
            "bedrooms": 2,
            "washrooms": 1,
            "has_garage": False,
            "has_attachedwashroom": False
        }

    # clamp / fix data
    bedrooms = data.get("bedrooms", 2)
    if bedrooms not in [1,2,3]:
        bedrooms = 2
    washrooms = data.get("washrooms", 1)
    if washrooms not in [1,2,3]:
        washrooms = 1
    has_garage = bool(data.get("has_garage", False))
    has_attachedwashroom = bool(data.get("has_attachedwashroom", False))

    return {
        "bedrooms": bedrooms,
        "washrooms": washrooms,
        "has_garage": has_garage,
        "has_attachedwashroom": has_attachedwashroom,
    }


def run_ground_floor_pipeline(bedrooms, washrooms, has_garage, has_attachedwashroom):
    """
    This replicates your old main.py pipeline that ends with 3 final ground-floor
    plans in 'pretty' (plan1.png/json, plan2.png/json, plan3.png/json).
    """
    # 1) Build rooms
    rooms = []
    for b in range(1, bedrooms + 1):
        rooms.append(f"Bedroom_{b}")
    for w in range(1, washrooms + 1):
        rooms.append(f"Washroom_{w}")
    rooms.append("Kitchen")
    if has_garage:
        rooms.append("Garage")

    # 2) Generate & visualize many floorplans (PNG + JSON)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    generator = FloorplanGenerator(rooms=rooms, attached_washroom=has_attachedwashroom)
    visualizer = FloorplanVisualizer()

    num_floorplans = 40
    for i in range(num_floorplans):
        fp_dict = generator.genetic_algorithm()
        base_name = f"floorplan_{i+1}"
        png_path = os.path.join(output_dir, base_name + ".png")
        json_path = os.path.join(output_dir, base_name + ".json")

        # Save PNG
        visualizer.plot_with_boundaries(
            fp_dict,
            png_path,
            FloorplanGenerator.FLOORPLAN_WIDTH,
            FloorplanGenerator.FLOORPLAN_HEIGHT
        )
        # Save JSON
        with open(json_path, "w") as jf:
            json.dump(fp_dict, jf)

    # 3) Detect & label living rooms -> finaloutput
    detector = RoomTypeDetector(input_dir="output", output_dir="finaloutput")
    detector.detect_and_label_images()

    # Copy matching JSON for each PNG in finaloutput
    final_out_pngs = [f for f in os.listdir("finaloutput") if f.lower().endswith(".png")]
    for pngf in final_out_pngs:
        copy_json_for_png(pngf, "output", "finaloutput")

    # 4) PerfectPlanSelector -> picks 3 => 'perfect'
    selector = PerfectPlanSelector(input_dir="finaloutput", output_dir="perfect")
    selector.select_connected_plans()

    # Copy JSON for those 3 perfect images
    perfect_pngs = [f for f in os.listdir("perfect") if f.lower().endswith(".png")]
    for pngf in perfect_pngs:
        copy_json_for_png(pngf, "finaloutput", "perfect")

    # 5) Make them pretty -> 'pretty'
    maker = PrettyFloorplanMaker(input_dir="perfect", output_dir="pretty")
    maker.make_pretty_floorplans()

    # Copy JSON for the final pretty images
    pretty_pngs = [f for f in os.listdir("pretty") if f.lower().endswith(".png")]
    for pngf in pretty_pngs:
        copy_json_for_png(pngf, "perfect", "pretty")

    # rename final 3 => plan1, plan2, plan3
    pretty_pngs = sorted(
        f for f in os.listdir("pretty")
        if f.lower().endswith(".png")
    )
    for i, old_png in enumerate(pretty_pngs[:3], start=1):
        new_png = f"plan{i}.png"
        rename_png_and_json(old_png, "pretty", new_png, "pretty")

    print("\nFinal 3 ground-floor plans => 'pretty/plan1.png/json', 'plan2.png/json', 'plan3.png/json'.\n")


def generate_first_floor_plans():
    """
    This replicates the final portion of your old main.py that asks:
      "Do you want to generate the 1st-floor plan from plan1, plan2, or plan3?"
    Then uses the chosen plan to create 3 new first-floor images in 'output_floor1'.
    """
    ans = input("Do you want to generate the 1st-floor plan from plan1, plan2, or plan3? (y/n): ").strip().lower()
    if ans == 'y':
        while True:
            try:
                choice = int(input("Pick which plan (1, 2, or 3): "))
                if choice not in [1,2,3]:
                    raise ValueError
                break
            except ValueError:
                print("Invalid choice. Must be 1, 2, or 3.")

        chosen_json = os.path.join("pretty", f"plan{choice}.json")
        if not os.path.exists(chosen_json):
            print(f"ERROR: Missing {chosen_json}, can't proceed.")
        else:
            with open(chosen_json, "r") as f:
                chosen_floorplan_dict = json.load(f)

            # Clear old files in output_floor1
            floor1_dir = "output_floor1"
            if os.path.exists(floor1_dir):
                for oldf in os.listdir(floor1_dir):
                    if oldf.lower().endswith(".png") or oldf.lower().endswith(".json"):
                        os.remove(os.path.join(floor1_dir, oldf))
            else:
                os.makedirs(floor1_dir, exist_ok=True)

            # We want 3 FIRST-FLOOR IMAGES, each with a DIFFERENT approach (1,2,3).
            approach_list = [1, 2, 3]
            # Reuse the same FloorplanVisualizer as before
            visualizer = FloorplanVisualizer()

            from floorplan_generator import FloorplanGenerator
            ff_width = FloorplanGenerator.FLOORPLAN_WIDTH
            ff_height = FloorplanGenerator.FLOORPLAN_HEIGHT

            for app_idx in approach_list:
                ff_gen = FirstFloorPlanGenerator(
                    chosen_floorplan_dict,
                    ff_width,
                    ff_height
                )
                # Generate a distinct approach
                first_floor_dict = ff_gen.generate_first_floor_plan(approach=app_idx)

                outbase = f"first_floor_plan_{choice}_approach{app_idx}"
                out_png = os.path.join(floor1_dir, outbase + ".png")
                out_json = os.path.join(floor1_dir, outbase + ".json")

                visualizer.plot_with_boundaries(
                    first_floor_dict,
                    out_png,
                    ff_width,
                    ff_height
                )
                with open(out_json, "w") as jf:
                    json.dump(first_floor_dict, jf)

            print(f"\nSaved 3 distinct first-floor plans (Approach #1, #2, #3) in '{floor1_dir}'.\n")


def main():
    print("Welcome! Please describe the floorplan you want (in plain English).")
    user_text = input("Your description: ")  # e.g. "I want 3 bedrooms, 2 washrooms, and a garage"

    # 1) Parse user_text with OpenAI
    specs = parse_floorplan_request(user_text)
    print("\nParsed specs =>", specs)

    # 2) Run the ground-floor pipeline => 'pretty/plan1, plan2, plan3'
    run_ground_floor_pipeline(
        bedrooms=specs["bedrooms"],
        washrooms=specs["washrooms"],
        has_garage=specs["has_garage"],
        has_attachedwashroom=specs["has_attachedwashroom"]
    )

    # 3) Optionally generate the 1st-floor plan
    generate_first_floor_plans()

    print("All done!")


if __name__ == "__main__":
    main()
