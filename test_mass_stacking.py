import argparse
import itertools
import json
import random
import re
import time
import os
import sys
from typing import List
from datetime import datetime
from vlm import VLM, AnthropicVLM, OpenAIVLM

"""
The purpose of this script is to evaluate VLM's ability to interact/query physical properties that is initially unknown.
Given the initial image and task prompt, the VLM can query the JSON file to extract/discover the unknown properties.
The VLM can adapt its plan based on the discovered properties to complete the task.
The VLM's output is the final plan to complete the task.    

Input: image of the scene, JSON file containing all object properties, task prompt
Output: plan to complete the task.

Evaluate on correctness, path length, and efficiency of the plan.
"""

JSON_FILE = "./sim_environment/outputs/03-26-16-25/environment_objects.json"

##############################
# Helper Functions
##############################
def load_world_state(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def query_mass(box_name, world_state):
    for obj in world_state:
        if obj["name"] == box_name:
            return obj["mass"]
    return None


def execute_action_sequence(action_seq, world_state):
    observations = []
    for action in action_seq:
        if action["action"] == "WEIGH":
            mass = query_mass(action["name"], world_state)
            action["observed_mass"] = mass
            observations.append(action)
    return observations

def clean_json_response(text):
    # Remove markdown code blocks and leading/trailing whitespace
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


##############################
# Main Function
##############################
def main():
    parser = argparse.ArgumentParser(description="Automated VLM Testing")
    parser.add_argument(
        "--provider", "-p", choices=["openai", "anthropic"], required=True, help="Choose VLM provider")
    parser.add_argument(
        "--num_tests", "-n", type=int, default=10, help="Number of tests to run (default: 10)")
    parser.add_argument(
        "--log_path", "-l", default="./data/logs/", help="Path to save test log (default: ./data/logs/)")

    args = parser.parse_args()

    img_path = "./sim_environment/outputs/03-26-16-25/camera_views_grid.png"
    world_state = load_world_state(JSON_FILE)

    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%m-%d-%H-%M")
    log_dir = f"./logs/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "vlm_final_response.json")

    system_prompt = """
    You are a robotic agent tasked with stacking boxes into a stable structure. 
    The boxes vary in mass, volume, and color, but the mass is hidden and must be discovered through interaction.
    Your goal is to minimize the number of WEIGH actions by discovering patterns 
    that relate visible attributes (e.g., color, volume) to mass. 

    ## Available Skills:
    - WEIGH: Check the mass of a specific box.
    - PICK_UP: Pick up a box.
    - PLACE: Place the box on the target stacking area.
    - MOVE: Move a box to another location on the table.

    ## Box attributes:
    `name` (box_0, box_1, ...), 
    `mass` (unknown),
    `color` (red, green, blue), 
    `volume` (small, medium, large), 
    and position.

    ## Output Format:
    You must output a structured list of actions in JSON format. 
    Each action should be clearly described. If you make an observation, include it. Example structure:

    ```json
    {
    "task": "Stack boxes by mass onto the yellow target area",
    "discovered_pattern": "Mass is determined by volume: small=1kg, medium=3kg, large=5kg",
    "action_sequence": [
        {
        "action": "WEIGH",
        "name": box_0,
        "color": "red",
        "volume": "small",
        "observed_mass": 1
        },
        {
        "action": "PICK_UP",
        "name": box_1,
        "color": "red",
        "volume": "large"
        },
        {
        "action": "PLACE",
        "target": "yellow_stack_area"
        },
        ...
    ],
    "note": "Only a few boxes were weighed to infer the mass-volume correlation. Remaining actions used inferred mass to reduce unnecessary interactions."
    }
    ```
    """


    test_images = [VLM.from_file(img_path)]
    vlm = OpenAIVLM(model="gpt-4o") if args.provider == "openai" else AnthropicVLM(
        model="claude-3-5-sonnet-20241022")



    test_prompt = f"""
    Verify that the given image is consistent with the described box scene. Then provide plan to stack the boxes according to mass (heaviest in bottom, lightest on top).
    Here is the current scene with 9 boxes of varying color, volume, and unknown mass. Begin by choosing boxes to weigh to uncover a potential pattern.
    
    Box Attributes:
    - box_0: red, small
    - box_1: blue, small
    - box_2: green, small
    - box_3: red, medium
    - box_4: blue, medium
    - box_5: green, medium
    - box_6: red, large
    - box_7: blue, large
    - box_8: green, large
    
    Respond with a JSON action sequence that starts the interaction.
    """


    conversation = [
        {"role": "user", "content": test_prompt}
    ]

    all_actions = []  # to store full action sequence across all iterations
    final_response = {}


    print(f" ***** Starting VLM Test *****")

    for step in range(3):  # Limit to 3 planning loops
        # Compose prompt with system + conversation history
        prompt_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in conversation])
        print(f"🧠 Prompt:\n{prompt_text}")

        response = vlm.analyze(
            prompt=prompt_text,
            images=test_images,
            system_prompt=system_prompt
        )

        print(f"🤖 Assistant Response: {response}")

        # Record model output
        conversation.append({
            "role": "assistant",
            "content": response.text
        })

        try:
            cleaned = clean_json_response(response.text)
            parsed_response = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print("❌ Failed to parse model response as JSON:\n", response.text)
            print("Error:", e)
            break

        final_response = parsed_response  # keep latest response

        actions = parsed_response.get("action_sequence", [])
        observations = execute_action_sequence(actions, world_state)
        print(f"🔍 Observations: {observations}")

        all_actions.extend(observations)

        # Append user feedback with observations
        obs_text = json.dumps(observations, indent=2)
        conversation.append({
            "role": "user",
            "content": f"Here are the results of your WEIGH actions:\n{obs_text}\nContinue planning."
        })
        print(f"📝 User Feedback:\n{obs_text}")


    print("✅ Final VLM Plan:")
    print(json.dumps(final_response, indent=2))

    # Save final plan including all interactions
    final_response["full_action_sequence"] = all_actions
    with open(log_file_path, "w") as f:
        json.dump(final_response, f, indent=2)

   

if __name__ == "__main__":
    main()
