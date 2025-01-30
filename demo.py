from vlm import VLM, OpenAIVLM

# Initialize the VLM with OpenAI implementation
vlm = OpenAIVLM(model="gpt-4o-mini")

# Analyze URL images
images = [
    # VLM.from_url("https://example.com/image1.jpg"),
    VLM.from_file("./cat.png"),
]

# Or use local images
# local_image = VLM.from_file("path_to_your_image.jpg")

# Analyze images
response = vlm.analyze(
    prompt="What is in the image?",
    images=images,
)

print(response)
