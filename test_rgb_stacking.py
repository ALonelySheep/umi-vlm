import argparse
import itertools
import json
import random
import re
import time
import os
from typing import List

from vlm import VLM, AnthropicVLM, OpenAIVLM


def extract_json_from_response(response_text: str) -> dict:
    """Extracts JSON content from response."""
    match = re.search(r"```json\n(\{.*?\})\n```", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}
    return {}


def evaluate_response(response_text: str, expected_order: List[str]) -> dict:
    """Validates if the response follows the expected JSON format and correct stacking order."""
    response_json = extract_json_from_response(response_text)
    steps = response_json.get("steps", [])
    returned_order = [step.get("object", "")
                      for step in steps if step.get("action") == "PICK_UP"]

    expected_steps = [
        {"action": "PICK_UP", "object": expected_order[0]},
        {"action": "PLACE", "location": "target"},
        {"action": "PICK_UP", "object": expected_order[1]},
        {"action": "PLACE", "location": f"on_{expected_order[0]}"},
        {"action": "PICK_UP", "object": expected_order[2]},
        {"action": "PLACE", "location": f"on_{expected_order[1]}"}
    ]

    success = (steps == expected_steps)
    return {"success": success, "expected_order": expected_order, "returned_order": returned_order}


def main():
    parser = argparse.ArgumentParser(description="Automated VLM Testing")
    parser.add_argument(
        "--provider", "-p", choices=["openai", "anthropic"], required=True, help="Choose VLM provider")
    parser.add_argument(
        "--num_tests", "-n", type=int, default=100, help="Number of tests to run (default: 100)")
    parser.add_argument(
        "--log_path", "-l", default="./data/logs/", help="Path to save test log (default: ./data/logs/)")

    args = parser.parse_args()

    img_path = "./data/images/blenderRGBboxes/Y.png"
    system_prompt = """
    You are a robotic arm operator tasked with stacking colored boxes into a designated target area. 
    You must follow a precise sequence of actions to complete the stacking process. 
    The allowed actions are: PICK_UP, PLACE, and MOVE. You can only move one box at a time. 
    
    Your response must be in the following JSON format:
    ```json
    {
        "steps": [
            {"action": "PICK_UP", "object": "red"},
            {"action": "PLACE", "location": "target"},
            {"action": "PICK_UP", "object": "blue"},
            {"action": "PLACE", "location": "on_red"},
            {"action": "PICK_UP", "object": "green"},
            {"action": "PLACE", "location": "on_blue"}
        ]
    }
    ```
    """

    all_orders = list(itertools.permutations(["red", "blue", "green"]))

    test_images = [VLM.from_file(img_path)]
    vlm = OpenAIVLM(model="gpt-4o") if args.provider == "openai" else AnthropicVLM(
        model="claude-3-5-sonnet-20241022")

    log_data = []
    successes = 0
    failed_tests = []

    print(
        f"Starting {args.num_tests} tests with {args.provider.upper()} model...\n")

    for i in range(args.num_tests):
        stacking_order = random.choice(all_orders)
        test_prompt = f"""
        Describe the image first in detail. Then, stack the boxes in this precise order: {stacking_order}.
        Return your response in JSON format as specified in the system prompt.
        """

        print(
            f"Running test {i+1}/{args.num_tests} with order {stacking_order}...")
        response = vlm.analyze(
            prompt=test_prompt, images=test_images, system_prompt=system_prompt)
        result = evaluate_response(response.text, stacking_order)

        log_entry = {"test_id": i+1, "expected_order": result["expected_order"],
                     "returned_order": result["returned_order"], "success": result["success"], "response": response.text}
        log_data.append(log_entry)
        if result["success"]:
            successes += 1
        else:
            failed_tests.append(i+1)
        print(f"Test {i+1} {'✅PASSED' if result['success'] else '❌FAILED'}")
        print(f"Expected order: {result['expected_order']}")
        print(f"Returned order: {result['returned_order']}\n")

    summary = {
        "provider": args.provider,
        "model": vlm.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": args.num_tests,
        "successful_tests": successes,
        "failed_tests": len(failed_tests),
        "failed_test_ids": failed_tests
    }

    log_data.append({"summary": summary})

    # create log folder if it doesn't exist
    log_path = args.log_path
    os.makedirs(log_path, exist_ok=True
                )

    # log file naming convention example: rgb_stacking_openai_2025-03-02_13-03-50.json
    log_file_name = f"rgb_stacking_{args.provider}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"

    with open(os.path.join(log_path, log_file_name), "w") as f:
        json.dump(log_data, f, indent=4)

    print(
        f"\nTest Completed. Success Rate: {successes}/{args.num_tests} ({(successes/args.num_tests)*100:.2f}%)")
    print(f"Failed tests: {len(failed_tests)} - IDs: {failed_tests}")


if __name__ == "__main__":
    main()
