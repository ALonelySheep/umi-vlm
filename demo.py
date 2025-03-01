from vlm import VLM, OpenAIVLM, AnthropicVLM
import argparse


def main():
    # CLI for model selection only
    parser = argparse.ArgumentParser(description="Test VLM implementations")
    parser.add_argument(
        "--provider",
        "-p",
        choices=["openai", "anthropic"],
        # default="anthropic",
        required=True,
        help="Choose VLM provider",
    )
    args = parser.parse_args()

    img_path = "images/3-boxes-same-sizes.png"
    system_prompt = """You are a helpful assistant. You need to help the user using a robotic arm to solve a puzzle. You only have access to a single robotic arm, so remember to put down the object before picking up another one.
    The allowed actions include: 1. picking the object. 2. Sliding the object 3. Shaking the object. 4. Putting down the object. 
    You can only use these actions to interact with the objects. You can collect sensory data during the process.
    We also have a set of sensors: 
    1. Microphone: Can be used to predict the state inside the object.
    2. Force/Torque sensor: Can be used to detect mass and torque for a specific object.
    """
    test_prompt = """
    What do you see in the image? If we want to sort the objects by empty/not_empty. What action should we take for each object? 
    State your reasoning step by step. If you are not sure about the action, please design a plan to collect more data using all the sensor you have access to.
    Format your response as JSON. 
    Here is a history of your previous actions:
    [timestamp 1]
    The red box is empty, the blue box is full, the green box is full.

    Please give your next action for sorting the object.
    """

    # Test configuration
    test_images = [
        VLM.from_file(img_path),
        # VLM.from_url("https://example.com/image1.jpg"),  # Uncomment to test URL
    ]

    try:
        # Initialize VLM based on provider
        if args.provider == "anthropic":
            vlm = AnthropicVLM(model="claude-3-5-sonnet-20241022")
        else:
            vlm = OpenAIVLM(model="gpt-4o")

        # Run analysis
        print(f"\nUsing provider: {args.provider}")
        print("\nAnalyzing images...")

        response = vlm.analyze(
            prompt=test_prompt, images=test_images, system_prompt=system_prompt
        )

        print("\nResponse:")
        print(response.text)

    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
