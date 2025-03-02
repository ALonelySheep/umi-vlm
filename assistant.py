import openai
import time
import os
import json

# Custom mock functions for assistant function calling


def get_current_temperature(location, unit):
    """Mock function to get the current temperature for a specific location."""
    return f"The temperature in {location} is 22 degrees {unit}."


def get_rain_probability(location):
    """Mock function to get the probability of rain for a specific location."""
    return f"The probability of rain in {location} is 10%."


# Initialize a single OpenAI client instance
client = openai.OpenAI()


def create_assistant(system_prompt="You are a helpful assistant."):
    """Creates an assistant with a configurable system prompt."""
    try:
        assistant = client.beta.assistants.create(
            name="Math Tutor",
            instructions=system_prompt,
            tools=[
                {"type": "code_interpreter"},
                # Adding custom function calling tools
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_temperature",
                        "description": "Get the current temperature for a specific location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "The city and state, e.g., San Francisco, CA"},
                                "unit": {"type": "string", "enum": ["Celsius", "Fahrenheit"], "description": "The temperature unit to use."}
                            },
                            "required": ["location", "unit"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_rain_probability",
                        "description": "Get the probability of rain for a specific location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "The city and state, e.g., San Francisco, CA"}
                            },
                            "required": ["location"]
                        }
                    }
                }

            ],
            model="gpt-4o-mini"
        )
        print("Assistant created:", assistant.id)
        return assistant.id
    except Exception as e:
        print("Error creating assistant:", str(e))
        return None


def get_or_create_thread(existing_thread_id=None):
    """Retrieves or creates a new thread."""
    if existing_thread_id:
        print("Using existing thread:", existing_thread_id)
        return existing_thread_id

    try:
        thread = client.beta.threads.create()
        print("New thread created:", thread.id)
        return thread.id
    except Exception as e:
        print("Error creating thread:", str(e))
        return None


def upload_image(image_path):
    """Uploads an image to OpenAI's file API for vision processing."""
    try:
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found.")
            return None

        with open(image_path, "rb") as image_file:
            uploaded_file = client.files.create(
                file=image_file,
                purpose="assistants"
            )
        print(f"Image uploaded successfully. File ID: {uploaded_file.id}")
        return uploaded_file.id
    except Exception as e:
        print("Error uploading image:", str(e))
        return None


def add_message_to_thread(thread_id, message, image_path=None, image_quality="auto"):
    """Adds a user message to the thread, optionally including an uploaded image."""
    try:
        message_content = [
            {"type": "text", "text": message}
        ]

        if image_path:
            file_id = upload_image(image_path)
            if file_id:
                message_content.append({
                    "type": "image_file",
                    "image_file": {"file_id": file_id}
                })

        response = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message_content
        )

        if not response:
            print("Error: Message was not successfully sent to the API.")
        else:
            print("Message successfully added to thread.")
    except Exception as e:
        print("Error adding message to thread:", str(e))


def run_assistant(thread_id, assistant_id):
    """Runs the assistant and handles different run states."""
    try:
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
            instructions="Please address the user as Jane Doe. The user has a premium account."
        )

        if run.status == "requires_action":
            tool_outputs = []
            print(f"Run requires action: {run.required_action}")
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                if function_name == "get_current_temperature":
                    output = get_current_temperature(
                        arguments["location"], arguments["unit"])
                elif function_name == "get_rain_probability":
                    output = get_rain_probability(arguments["location"])
                else:
                    output = "Unknown function call"

                tool_outputs.append(
                    {"tool_call_id": tool_call.id, "output": output})

            run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )

        if run.status == "completed":
            print("Run completed successfully.")
        else:
            print("Run ended with status:", run.status)

        return run
    except Exception as e:
        print("Error running assistant:", str(e))
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
                print("Assistant:", response_text)
                return response_text
    except Exception as e:
        print("Error retrieving latest response:", str(e))


def multi_round_conversation(assistant_id, thread_id):
    """Handles a multi-turn conversation."""
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Ending conversation.")
            break
        add_message_to_thread(
            thread_id, user_input, image_path='./data/images/cat.jpg', image_quality="auto")
        run_assistant(thread_id, assistant_id)
        get_latest_response(thread_id)


def main():
    """Main execution flow."""
    system_prompt = "You are a knowledgeable assistant. Answer queries concisely."
    assistant_id = create_assistant(system_prompt)
    if not assistant_id:
        return

    thread_id = None  # Keep thread_id in memory instead of writing to a file
    thread_id = get_or_create_thread(thread_id)
    if not thread_id:
        return

    multi_round_conversation(assistant_id, thread_id)

    try:
        client.beta.assistants.delete(assistant_id)
        print("Assistant deleted.")
    except Exception as e:
        print("Error deleting assistant:", str(e))


if __name__ == "__main__":
    main()
