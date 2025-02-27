from vlm import VLM, OpenAIVLM, AnthropicVLM
import argparse
import librosa
import sys
import json
from scipy.signal import spectrogram
import numpy as np

def extract_audio_features_from_dir(audio_dir):
    """
    given directory, return list of audio features
    """
    audio_features = {}
    # audio_file_names = ["Paper.m4a", "Plastic.m4a", "Fabric.m4a"]
    audio_file_names = ["GX010190.wav", "GX010191.wav", "GX010192.wav", "GX010193.wav", "GX010194.wav"]
    # [N, N, N, e, e]
    audio_data = [] #store audio data

    #Iterate through audio files names to load ".m4a" files
    for audio_file_name in audio_file_names:
        audio_file_path = f"{audio_dir}/{audio_file_name}"
        #load audio file given path using librosa library (newer version)
        y_audio, sample_rate = librosa.load(audio_file_path, sr=None)

        audio_data.append(y_audio)

    #extract features from audio data including [Spectral Centroid, spectral bandwidth, spectral flatness, zero crossing rate, RMS energy]
    for i, audio in enumerate(audio_data):
            

        # Convert the stereo audio to mono by averaging the two channels (if needed)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Calculate the root mean square (RMS) energy of the mono audio signal
        rms_energy = np.sqrt(np.mean(audio**2))

        # Perform a simple frequency analysis using the Fast Fourier Transform (FFT)
        fft_spectrum_3 = np.fft.fft(audio)
        frequencies_3 = np.fft.fftfreq(len(fft_spectrum_3), 1 / sample_rate)
        magnitude_3 = np.abs(fft_spectrum_3)

        # Calculate the spectral centroid (a rough indicator of the "brightness" of the sound)
        spectral_centroid = np.sum(frequencies_3 * magnitude_3) / np.sum(magnitude_3)

        # Calculate zero-crossing rate (a measure of how often the signal changes sign)
        zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(audio))))

        rms_energy, spectral_centroid, zero_crossing_rate

        # Store extracted audio features into a dictionary for each audio file
        audio_features[f'audio_{i}'] = {
            "spectral_centroid": float(spectral_centroid),
            "zero_crossing_rate": float(zero_crossing_rate),
            "rms_energy": float(rms_energy)
        }

    return audio_features

def main():

    #path to audio files
    audio_dir = "/home/mark/github/umi-vlm"
    audio_features = extract_audio_features_from_dir(audio_dir)

    print(json.dumps(audio_features, indent=4))
    # sys.exit(0)

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
    system_prompt = "You are a helpful assistant with expertise in recognizing patterns and identifying classes based on visual representations of audio data."
    # system_prompt = "You are a helpful assistant. You need to help the user to solve a problem."

    
    test_prompt = '''
Given the audio features extracted from the audio files, which include spectral centroid, spectral bandwidth, 
spectral flatness, zero crossing rate, and RMS energy. The audio clips are recordings of hand shaking the boxes, please classify the audio files into empty boxes and not empty boxes.
Here is the audio features extracted from the audio files:
'''
#     test_prompt = '''
# Your task is to analyze spectrograms, which are visual representations of the frequency spectrum of sound over time, and to determine the most likely sound class for a given spectrogram.

# '''

    # convert dict to string
    text_audio_features = json.dumps(audio_features, indent=4)
    test_prompt += text_audio_features
    print(test_prompt)

    # sys.exit(0)

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