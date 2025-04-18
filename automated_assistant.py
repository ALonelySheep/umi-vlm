import time
import uuid
import openai
import json
import os
import base64
import requests
import argparse
from PIL import Image
import io
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
SERVER_URL = "http://localhost:5000"

# Initialize OpenAI client
client = openai.OpenAI()

# Helper functions from run_client.py for ISAAC Sim interaction


def get_status():
    """Get the server status"""
    try:
        response = requests.get(f"{SERVER_URL}/status")
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Could not connect to server. Make sure the server is running."}


def list_objects():
    """Get a list of all objects in the scene"""
    try:
        response = requests.get(f"{SERVER_URL}/objects")
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Could not connect to server. Make sure the server is running."}


def stack_objects(source_object, target_object):
    """Stack source object on top of target object - actual implementation instead of mock"""
    try:
        data = {
            "source": source_object,
            "target": target_object
        }
        response = requests.post(f"{SERVER_URL}/stack", json=data)
        print(
            f"🤖[Actual stacking] Stacking {source_object} on {target_object}.")
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Could not connect to server. Make sure the server is running."}


def capture_image(view="iso"):
    """Capture an image from the specified viewpoint (default: iso)"""
    try:
        response = requests.get(f"{SERVER_URL}/capture", params={"view": view})
        if response.status_code != 200:
            return {"status": "error", "message": f"Failed to capture image: {response.text}", "image_data": None}

        result = response.json()
        image_data = None
        filepath = None

        if "image_base64" in result:
            # Decode base64 image
            image_data = base64.b64decode(result["image_base64"])

            # Save locally for reference (optional)
            images_dir = os.path.join("data", "images", "captures")
            os.makedirs(images_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(images_dir, f"{view}_view_{timestamp}.png")

            with open(filepath, "wb") as f:
                f.write(image_data)

            print(f"Image captured from {view} view and saved to {filepath}")

        return {"status": "success", "message": "Image captured", "image_data": image_data, "filepath": filepath}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Could not connect to server. Make sure the server is running.", "image_data": None}
    except Exception as e:
        return {"status": "error", "message": f"Error capturing image: {str(e)}", "image_data": None}

# Modified assistant functions from assistant.py


def create_assistant(system_prompt="You are a helpful assistant."):
    """Creates an assistant with a configurable system prompt."""
    try:
        assistant = client.beta.assistants.create(
            name="Isaac Sim Robot Controller",
            instructions=system_prompt,
            tools=[
                {"type": "code_interpreter"},
                # Real stacking function instead of mock move_object
                {
                    "type": "function",
                    "function": {
                        "name": "stack_objects",
                        "description": "Use a robotic arm to stack one object on top of another.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "source_object": {
                                    "type": "string",
                                    "description": "The name of the object to pick up and stack."
                                },
                                "target_object": {
                                    "type": "string",
                                    "description": "The name of the object to stack the source object on."
                                }
                            },
                            "required": ["source_object", "target_object"]
                        }
                    }
                },
            ],
            model="gpt-4o"
        )
        print("✅Assistant created:", assistant.id)
        return assistant
    except Exception as e:
        print("❌Error creating assistant:", str(e))
        return None


def get_or_create_thread(existing_thread_id=None):
    """Retrieves or creates a new thread."""
    if existing_thread_id:
        print("📒Using existing thread:", existing_thread_id)
        return client.beta.threads.retrieve(existing_thread_id)

    try:
        thread = client.beta.threads.create()
        print("✅New thread created:", thread.id)
        return thread
    except Exception as e:
        print("❌Error creating thread:", str(e))
        return None


def add_message_to_thread(thread_id, message, image_path=None):
    """Adds a user message to the thread, optionally including an image."""
    print(f"📩 Adding message to thread {thread_id}: {message}")
    try:
        message_content = [
            {"type": "text", "text": message}
        ]

        # Upload image if provided
        if image_path:
            with open(image_path, "rb") as image_file:
                file = client.files.create(
                    file=image_file,
                    purpose="vision",
                )

                message_content.append({
                    "type": "image_file",
                    "image_file": {
                        "file_id": file.id,
                        "detail": "auto"
                    },
                })

        response = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message_content
        )

        if not response:
            print("❌Error: Message was not successfully sent to the API.")
        else:
            print("✅Message successfully added to thread.")
    except Exception as e:
        print("❌Error adding message to thread:", str(e))


def run_assistant(thread_id, assistant_id):
    """Runs the assistant and handles multiple sequential stack_objects calls in a single run."""
    try:
        # Start the run once
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )

        # Keep handling actions if assistant requires them
        while run.status == "requires_action":
            tool_outputs = []
            print(f"🤖 Run requires action: {run.required_action}")

            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                if function_name == "stack_objects":
                    source_object = arguments["source_object"]
                    target_object = arguments["target_object"]
                    result = stack_objects(source_object, target_object)
                    output = json.dumps(result)
                else:
                    output = "Unknown function call"

                tool_outputs.append(
                    {"tool_call_id": tool_call.id, "output": output}
                )

            # Submit tool outputs and poll again
            run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )

        # Run completed successfully
        if run.status == "completed":
            print("✅ Run completed successfully.")
        else:
            print(f"❌ Run ended with status: {run.status}")

        return run

    except Exception as e:
        print(f"❌ Error running assistant: {str(e)}")
        return None


def get_latest_response(thread_id):
    """Retrieves the latest assistant response."""
    try:
        messages = client.beta.threads.messages.list(
            thread_id=thread_id, order="desc", limit=1)
        if messages.data:
            latest_message = messages.data[0]
            if latest_message.role == "assistant":
                assert latest_message.content[0].type == "text"
                response_text = latest_message.content[0].text.value
                print("🤖 Assistant:", response_text)
                return response_text
        return None
    except Exception as e:
        print("❌Error retrieving latest response:", str(e))
        return None


def check_for_completion(response_text):
    """Check if the assistant indicates completion with the word 'finish'"""
    if response_text and "finish" in response_text.lower():
        return True
    return False


def save_conversation_log(conversation_log, thread_id):
    """Saves the conversation to a JSON file along with images."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Create directory for conversation
    logs_dir = os.path.join("data", "logs")
    conversation_dir = os.path.join(
        logs_dir, f"conversation_{thread_id}_{timestamp}")
    os.makedirs(conversation_dir, exist_ok=True)

    # Save images from the conversation
    image_paths_map = {}
    for idx, message in enumerate(conversation_log["messages"]):
        if message.get("image_path"):
            # Copy image to the conversation directory
            src_path = message["image_path"]
            img_filename = os.path.basename(src_path)
            # Add message index to filename to maintain order
            dst_filename = f"{idx:03d}_{img_filename}"
            dst_path = os.path.join(conversation_dir, dst_filename)

            try:
                shutil.copy2(src_path, dst_path)
                # Record the relative path for the JSON file
                rel_path = os.path.join("images", dst_filename)
                image_paths_map[src_path] = rel_path
                # Update the message with the new relative path
                message["saved_image"] = rel_path
            except Exception as e:
                print(f"⚠️ Could not copy image {src_path}: {str(e)}")

    # Save the JSON log
    json_filename = f"conversation_{thread_id}_{timestamp}.json"
    json_filepath = os.path.join(conversation_dir, json_filename)

    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(conversation_log, f, indent=2)

    print(f"✅ Conversation and images saved to {conversation_dir}")
    return conversation_dir


def get_valid_stacking_orders():
    """Returns a list of valid stacking orders for three colored boxes (R, G, B)"""
    colors = ['R', 'G', 'B']
    import itertools
    return [''.join(p) for p in itertools.permutations(colors)]


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run automated assistant with ISAAC Sim')
    parser.add_argument('--stacking-order', type=str, choices=get_valid_stacking_orders(),
                        help='Specify the stacking order of the 3 colored boxes (top to bottom). Examples: RGB, RBG, BGR')
    args = parser.parse_args()

    # Check if Isaac Sim server is running
    status = get_status()
    if status.get("status") == "error":
        print("❌ Error: Cannot connect to Isaac Sim server.")
        return

    print("✅ Connected to Isaac Sim server.")

    # Get list of objects for system prompt
    objects_result = list_objects()
    objects_list = objects_result.get("objects", [])
    objects_str = ", ".join(
        objects_list) if objects_list else "No objects found"

    # Create system prompt with objects
    system_prompt = f"""
    You are a helper system that can use a robotic arm to manipulate objects inside a simulated environment.
    You can only perform one operation at a time.
    
    Available objects in the environment: {objects_str}
    
    After each step, you will receive images for you to devide if the operation was successful or not.
    Based on this feedback:
    - If successful, continue with the next step
    - If unsuccessful, try an alternative approach
    """

    # Add stacking order instruction if provided
    if args.stacking_order:
        system_prompt += f"""
    
    Your goal is to stack the colored cubes in this specific order from top to bottom: {args.stacking_order}.
    For example, if the order is RGB, the Red cube should be on top, followed by Green in the middle, and Blue at the bottom.
    """
    else:
        system_prompt += """
    
    Analyze the environment and propose a stacking arrangement that you think would be interesting.
    """

    system_prompt += """
    
    When you believe the entire task is complete, say "FINISH" clearly in your response.
    
    Remember:
    1. Analyze the image to understand the current state of objects
    2. Plan your actions carefully
    3. Confirm each action worked before proceeding
    4. When the task is complete, say "FINISH"
    """

    # Create assistant with this system prompt
    assistant = create_assistant(system_prompt)
    if not assistant:
        print("❌ Error creating assistant.")
        return

    # Create a new thread
    thread = get_or_create_thread()
    if not thread:
        print("❌ Error creating thread.")
        return

    # Create a timestamp for this run
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Track the conversation for logging
    conversation_log = {
        "timestamp_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt": system_prompt,
        "stacking_order": args.stacking_order if args.stacking_order else "Not specified",
        "run_id": run_timestamp,
        "messages": []
    }

    # Prepare initial prompt based on stacking order
    if args.stacking_order:
        initial_prompt = f"Here is the current state of the environment. Please start stacking the colored cubes in the order {args.stacking_order} (from top to bottom)."
    else:
        initial_prompt = "Here is the current state of the environment. What would you like to do with the objects you see? Please describe your plan."

    # Start the process - get first image
    image_result = capture_image(view="iso")
    if image_result["status"] == "error" or not image_result["image_data"]:
        print("❌ Error capturing initial image.")
        return

    # Start the conversation with initial prompt and image
    add_message_to_thread(thread.id, initial_prompt, image_result["filepath"])

    # Add to conversation log with image path
    conversation_log["messages"].append({
        "role": "user",
        "content": initial_prompt,
        "has_image": True,
        "image_path": image_result["filepath"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    # Run the assistant for the first time
    run = run_assistant(thread.id, assistant.id)
    if not run:
        print("❌ Error running assistant.")
        return

    # Get response
    response = get_latest_response(thread.id)
    if not response:
        print("❌ Error getting assistant response.")
        return

    # Add to conversation log
    conversation_log["messages"].append({
        "role": "assistant",
        "content": response,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    # Main conversation loop
    conversation_finished = check_for_completion(response)
    while not conversation_finished:
        # Get new image after action
        image_result = capture_image(view="iso")
        if image_result["status"] == "error" or not image_result["image_data"]:
            print("❌ Error capturing updated image.")
            break

        # Ask for confirmation and next step
        confirm_message = "Here is the current state after your last action. Was your operation successful?"

        # Add stacking order reminder if it was specified
        if args.stacking_order:
            confirm_message += f" Remember, your goal is to stack the cubes in the order {args.stacking_order} (from top to bottom)."

        confirm_message += " Please confirm and continue with your next action, or say 'FINISH' if you're done."

        add_message_to_thread(thread.id, confirm_message,
                              image_result["filepath"])

        # Add to conversation log with image path
        conversation_log["messages"].append({
            "role": "user",
            "content": confirm_message,
            "has_image": True,
            "image_path": image_result["filepath"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        # Run the assistant
        run = run_assistant(thread.id, assistant.id)
        if not run:
            print("❌ Error running assistant.")
            break

        # Get response
        response = get_latest_response(thread.id)
        if not response:
            print("❌ Error getting assistant response.")
            break

        # Add to conversation log
        conversation_log["messages"].append({
            "role": "assistant",
            "content": response,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        # Check if conversation is finished
        conversation_finished = check_for_completion(response)

        # Small delay to avoid API rate limits
        time.sleep(1)

    # Conversation ended
    conversation_log["timestamp_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print("✅ Conversation completed!")

    # Save conversation and images
    saved_dir = save_conversation_log(conversation_log, thread.id)
    print(f"✅ Saved conversation and images to: {saved_dir}")


if __name__ == "__main__":
    main()
