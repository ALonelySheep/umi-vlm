import openai


def create_assistant():
    client = openai.OpenAI()

    assistant = client.beta.assistants.create(
        name="My Minimal Assistant",
        instructions="You are a helpful assistant.",
        model="gpt-4o-mini"
    )

    print("Assistant created:", assistant.id)
    return assistant.id


def create_thread():
    client = openai.OpenAI()

    thread = client.beta.threads.create()
    print("Thread created:", thread.id)
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

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )
    print("Run started:", run.id)
    return run.id


def main():
    assistant_id = create_assistant()
    thread_id = create_thread()
    add_message_to_thread(thread_id, "Hello, how can you help me?")
    run_assistant(thread_id, assistant_id)


if __name__ == "__main__":
    main()
