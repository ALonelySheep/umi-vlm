import time  # For timestamps
import uuid  # For unique session ID
import uuid  # <-- Import for generating unique keys
import openai
import time
import os
import json
import streamlit as st
from PIL import Image
import io  # <-- Import for handling in-memory byte streams

# Custom mock functions for assistant function calling
# ! NOTE: the request will time out after roughly 10 mins,
# ! This might cause a problem when using diffusion policy


def move_object(cube_color):
    """Mock function to move an object of a specific color."""
    print(f"🤖[Simulating...] Moving the {cube_color} cube.")
    return f"Moving the {cube_color} cube."


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
                # See https://platform.openai.com/docs/guides/structured-outputs#examples
                # for details on JSON schema
                {
                    "type": "function",
                    "function": {
                        "name": "move_object",
                        "description": "Use a robotic arm to move an object.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "cube_color": {
                                    "type": "string",
                                    "enum": ["red", "green", "blue"],
                                    "description": "The color of the cube to move."
                                }
                            },
                            "required": ["cube_color"]
                        }
                    }
                },
            ],
            model="gpt-4o-mini"
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
        return existing_thread_id

    try:
        thread = client.beta.threads.create()
        print("✅New thread created:", thread.id)
        return thread
    except Exception as e:
        print("❌Error creating thread:", str(e))
        return None


def add_message_to_thread(thread_id, message, image_id=None, detail="auto"):
    """Adds a user message to the thread, optionally including an uploaded image."""
    try:
        message_content = [
            {"type": "text", "text": message}
        ]

        if image_id:
            message_content.append({
                "type": "image_file",
                "image_file": {
                    "file_id": image_id,
                    "detail": detail
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
    """Runs the assistant and handles different run states."""
    try:
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
            instructions="Please address the user as Jane Doe. The user has a premium account."
        )

        if run.status == "requires_action":
            tool_outputs = []
            print(f"🤖Run requires action: {run.required_action}")
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                if function_name == "move_object":
                    output = move_object(arguments["cube_color"])
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
            print("✅Run completed successfully.")
        else:
            print("❌Run ended with status:", run.status)

        return run
    except Exception as e:
        print("❌Error running assistant:", str(e))
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
                print("🐤Assistant:", response_text)
                return response_text
    except Exception as e:
        print("❌Error retrieving latest response:", str(e))


# def upload_image(image_path):
#     """Uploads an image to OpenAI's file API for vision processing."""
#     try:
#         if not os.path.exists(image_path):
#             print(f"Error: File '{image_path}' not found.")
#             return None

#         with open(image_path, "rb") as image_file:
#             uploaded_file = client.files.create(
#                 file=image_file,
#                 purpose="assistants"
#             )
#         print(f"Image uploaded successfully. File ID: {uploaded_file.id}")
#         return uploaded_file.id
#     except Exception as e:
#         print("Error uploading image:", str(e))
#         return None


def upload_image_from_memory(uploaded_file):
    """Uploads an image to OpenAI from memory with a proper filename."""
    try:
        # Get file extension
        # Extracts .png, .jpg, etc.
        file_extension = os.path.splitext(uploaded_file.name)[-1]

        # Ensure the file has a valid extension (default to .png if unknown)
        if file_extension.lower() not in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
            file_extension = ".png"

        # Convert file object to a byte stream
        image_bytes = uploaded_file.getvalue()

        # Create a file-like object with a correct filename
        file_like_object = io.BytesIO(image_bytes)
        # Set a valid filename
        file_like_object.name = f"uploaded_image{file_extension}"

        # Upload directly from memory with correct filename
        uploaded_file = client.files.create(
            file=file_like_object,
            purpose="assistants"
        )

        print(f"🟦Image uploaded successfully. File ID: {uploaded_file.id}")
        return uploaded_file.id

    except Exception as e:
        print("❌Error uploading image:", str(e))
        return None


# === Streamlit GUI Integration ===
# import streamlit as st
# from PIL import Image

# === Streamlit GUI Integration ===


# === Streamlit GUI Integration ===

# === Streamlit GUI Integration ===

def chatbot_ui():
    """Streamlit-based graphical user interface for interacting with the assistant."""
    st.set_page_config(page_title="AI Chatbot", layout="wide")  # Set UI layout

    st.title("🤖 AI Assistant Chatbot")

    # Set up session state for storing conversation history and metadata
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}]

    # Fix: Track file uploader key to force reset
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = str(
            uuid.uuid4())  # Generate a random key

    # Initialize metadata if not set
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())  # Unique session ID

    if "timestamp" not in st.session_state:
        st.session_state["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Use full objects instead of just IDs
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = create_assistant(
            "You are a knowledgeable assistant.")

    if "thread" not in st.session_state:
        st.session_state["thread"] = get_or_create_thread()

    # === Sidebar for Saving Chat ===
    with st.sidebar:
        st.header("Chat Options")
        if st.button("💾 Save Chat Log"):
            save_chat_to_json()

    # === Display chat history (Newest at the BOTTOM) ===
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image" in msg:
                # Show actual filename
                image_filename = msg.get("image_name", "Uploaded Image")
                st.image(msg["image"], caption=image_filename,
                         use_container_width=True)

    # === User input ===
    user_input = st.chat_input("Type your message...")

    # === Image upload ===
    uploaded_image = st.file_uploader(
        "Upload an image (optional)", type=["png", "jpg", "jpeg"], key=st.session_state["file_uploader_key"]
    )

    if user_input:
        if not user_input.strip():
            st.warning("Please enter a message before sending.")
        else:
            # Handle image upload properly
            file_id = None
            image_filename = None  # Track the actual filename

            if uploaded_image:
                file_id = upload_image_from_memory(
                    uploaded_image)  # Upload from memory
                image_filename = uploaded_image.name  # Extract actual filename

            # Store user message
            user_message = {"role": "user", "content": user_input}
            if uploaded_image:
                # Store the uploaded image
                user_message["image"] = uploaded_image
                # Store actual filename
                user_message["image_name"] = image_filename

            st.session_state["messages"].append(user_message)

            # Display user input in chat
            with st.chat_message("user"):
                st.markdown(user_input)
                if uploaded_image:
                    image = Image.open(uploaded_image)
                    st.image(image, caption=image_filename,
                             use_container_width=True)  # Show actual filename

            # Send message & image to assistant
            add_message_to_thread(
                st.session_state["thread"].id, user_input, image_id=file_id, detail="auto"
            )

            # Run assistant
            run_assistant(
                st.session_state["thread"].id, st.session_state["assistant"].id)
            response_text = get_latest_response(st.session_state["thread"].id)

            # Store and display assistant response
            assistant_message = {"role": "assistant", "content": response_text}
            st.session_state["messages"].append(assistant_message)

            with st.chat_message("assistant"):
                st.markdown(response_text)

            # Fix: Generate a new key to reset the file uploader
            st.session_state["file_uploader_key"] = str(uuid.uuid4())

            # Refresh UI for smooth interaction
            st.rerun()


# === Save Chat to JSON File ===
def save_chat_to_json():
    """Saves the current chat session to a JSON file, including metadata."""

    # Function to process non-serializable objects
    def serialize_object(obj):
        """Convert objects into readable formats for JSON compatibility."""
        if isinstance(obj, list):  # Handle lists of tools
            return [serialize_object(item) for item in obj]
        elif hasattr(obj, '__dict__'):  # Convert objects to dictionary safely
            obj_dict = obj.__dict__.copy()
            # Remove tools (they contain non-serializable objects)
            obj_dict.pop('tools', None)
            # Exclude tool resources (not serializable)
            obj_dict.pop('tool_resources', None)
            # Exclude empty metadata if not needed
            obj_dict.pop('metadata', None)
            return obj_dict
        elif isinstance(obj, (str, int, float, bool, type(None))):  # Keep primitive types
            return obj
        else:
            return str(obj)  # Convert other objects to string

    # Convert assistant and thread to a JSON-safe format
    assistant_data = serialize_object(st.session_state["assistant"])
    assistant_data["tools"] = [str(
        tool) for tool in st.session_state["assistant"].tools]  # Convert tools to string
    thread_data = serialize_object(st.session_state["thread"])

    chat_log = {
        "timestamp": st.session_state["timestamp"],
        # Store full assistant metadata (without non-serializable fields)
        "assistant": assistant_data,
        "thread": thread_data,  # Store full thread metadata
        "messages": []
    }

    # Convert messages into a savable format
    for msg in st.session_state["messages"]:
        formatted_msg = {
            "role": msg["role"],
            "content": msg["content"]
        }
        if "image_name" in msg:
            # Save image filename
            formatted_msg["image_name"] = msg["image_name"]
        chat_log["messages"].append(formatted_msg)

    # Generate filename using timestamp instead of session_id
    timestamp_str = st.session_state["timestamp"].replace(
        ":", "-").replace(" ", "_")
    filename = f"chat_log_{timestamp_str}.json"
    # save in subdirectory './data/logs'
    os.makedirs("./data/logs", exist_ok=True)
    # filepath = os.path.join(os.getcwd(), filename)
    filepath = os.path.join("./data/logs", filename)

    # Write JSON log to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chat_log, f, indent=4)

    st.sidebar.success(f"Chat saved to {filepath}.")


# === Run the UI ===
if __name__ == "__main__":
    chatbot_ui()
