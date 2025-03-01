import openai
import time


def create_assistant():
    client = openai.OpenAI()

    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4o-mini"
    )

    print("Assistant created:", assistant.id)
    return assistant.id


def get_or_create_thread(existing_thread_id=None):
    client = openai.OpenAI()
    if existing_thread_id:
        print("Using existing thread:", existing_thread_id)
        return existing_thread_id

    thread = client.beta.threads.create()
    print("New thread created:", thread.id)
    return thread.id


def add_message_to_thread(thread_id, message):
    client = openai.OpenAI()

    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message
    )
    print("Message added to thread.")


def run_assistant(thread_id, assistant_id):
    client = openai.OpenAI()

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions="Please address the user as Jane Doe. The user has a premium account."
    )

    print("Run completed with status:", run.status)
    return run


def get_responses(thread_id):
    client = openai.OpenAI()

    messages = client.beta.threads.messages.list(thread_id=thread_id)
    print("Messages:")
    for message in messages:
        assert message.content[0].type == "text"
        print({"role": message.role, "message": message.content[0].text.value})


def multi_round_conversation(assistant_id, thread_id):
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Ending conversation.")
            break
        add_message_to_thread(thread_id, user_input)
        run_assistant(thread_id, assistant_id)
        get_responses(thread_id)


def main():
    assistant_id = create_assistant()
    thread_id = None  # Keep thread_id in memory instead of writing to a file
    thread_id = get_or_create_thread(thread_id)
    multi_round_conversation(assistant_id, thread_id)

    client = openai.OpenAI()
    client.beta.assistants.delete(assistant_id)
    print("Assistant deleted.")


if __name__ == "__main__":
    main()
