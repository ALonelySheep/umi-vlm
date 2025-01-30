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

    img_path = "images/3-boxes-different-sizes.png"
    system_prompt = "You are a helpful assistant. You need to help the user using a robotic arm to solve a puzzle."
    test_prompt = "What do you see in the image? If we need to stack the boxes, which box should we pick first? Please describe the box and its position in detail and format your response as JSON."

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
            vlm = OpenAIVLM(model="gpt-4o-mini")

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
