from vlm import VLM, OpenAIVLM, AnthropicVLM
import argparse
import librosa
import librosa.display
import sys
import json
from scipy.signal import spectrogram
import numpy as np
import base64
import matplotlib.pyplot as plt
import copy
import io

def load_audio_files(audio_dir):
    """
    Given directory, return list of audio files with separated reference and test data
    """
    # Define audio files by class
    audio_files_by_class = {
        'paper': ["GX010209.wav", "GX010210.wav", "GX010211.wav", "GX010212.wav", "GX010213.wav"],
        'plastic': ["GX010214.wav", "GX010215.wav", "GX010216.wav", "GX010217.wav", "GX010218.wav"],
        'fabric': ["GX010219.wav", "GX010220.wav", "GX010221.wav", "GX010222.wav", "GX010223.wav"]
    }

    # Combine all lists while maintaining class information
    audio_file_names = []
    for class_name, files in audio_files_by_class.items():
        audio_file_names.extend(files)

    audio_data = []  # store audio data
    
    # Load all audio files
    for audio_file_name in audio_file_names:
        audio_file_path = f"{audio_dir}/{audio_file_name}"
        y_audio, sample_rate = librosa.load(audio_file_path, sr=None)
        audio_data.append(y_audio)

    # Split into reference and test data by taking first sample of each class
    reference_audio_data = []
    test_audio_data = []
    
    current_idx = 0
    for files in audio_files_by_class.values():
        class_size = len(files)
        # First sample becomes reference
        reference_audio_data.append(audio_data[current_idx])
        # Remaining samples become test data
        test_audio_data.extend(audio_data[current_idx + 1:current_idx + class_size])
        current_idx += class_size

    return audio_data, sample_rate, reference_audio_data, test_audio_data

def extract_audio_feature(audio_data, sample_rate):
    """
    given audio data in list, return audio features in dict
    """
    audio_features = {}

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

def extract_spectrogram(audio_data, sample_rate):
    """
    Given audio data and sample rate, return spectrogram.
    Audio should be resampled to 22050 Hz then convert using STFT using 2048 window size and 512 hop length.
    Spectrogram should be linear frequency scale using viridis colormap and have axis label "Time" and "Frequency" but remove colormap scale.
    """

    spectrograms_data = []
    target_sr = 22050  # Target sample rate
    
    for audio in audio_data:
        # Resample audio to 22050 Hz if needed
        if sample_rate != target_sr:
            audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=target_sr)
        
        # Compute STFT (Short-Time Fourier Transform)
        D = librosa.stft(audio, n_fft=2048, hop_length=512)
        
        # Convert to magnitude spectrogram
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Store the spectrogram data instead of the figure
        spectrograms_data.append(D_db)
        
    return spectrograms_data

def visualize_reference_test_spectrograms(reference_spectrograms, test_spectrograms, sample_rate, plot_spectrograms=False):
    """
    Given reference and test spectrograms, create visualizations.
    Returns:
        - Reference grid figure (1x3)
        - List of individual test spectrogram figures
    """
    # Create figure for reference spectrograms (1x3 grid)
    fig_ref = plt.figure(figsize=(15, 5))
    
    # Plot reference spectrograms
    for i, ref_spec in enumerate(reference_spectrograms, 1):
        plt.subplot(1, 3, i)
        librosa.display.specshow(
            ref_spec,
            sr=sample_rate,
            x_axis='time',
            y_axis='linear',
            cmap='viridis'
        )
        plt.title(f'{["Plastic", "Fabric", "Paper"][i-1]}')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    
    # Create individual figures for test spectrograms
    test_figs = []
    for i, test_spec in enumerate(test_spectrograms):
        # Create new figure for each test spectrogram
        fig = plt.figure(figsize=(5, 5))
        
        librosa.display.specshow(
            test_spec,
            sr=sample_rate,
            x_axis='time',
            y_axis='linear',
            cmap='viridis'
        )
        
        # Determine class based on index
        class_idx = i // 4  # Every 4 samples belong to same class
        class_name = ["Plastic", "Fabric", "Paper"][class_idx]
        plt.title(f'Test {i+1}: {class_name}')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        test_figs.append(fig)
        
        # Close figure to free memory if not displaying
        if not plot_spectrograms:
            plt.close(fig)
    
    # Show figures if requested
    if plot_spectrograms:
        plt.show()
    else:
        plt.close(fig_ref)
    
    # Convert figures to base64 for VLM
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    
    # Convert reference figure to base64
    ref_base64 = fig_to_base64(fig_ref)
    
    # Convert test figures to base64
    test_base64_list = []
    for fig in test_figs:
        test_base64_list.append({
            "base64": fig_to_base64(fig),
            "media_type": "image/png"
        })
    
    # Return reference image and list of test images
    return (
        {"base64": ref_base64, "media_type": "image/png"},
        test_base64_list
    )


def main():

    parser = argparse.ArgumentParser(description="Test VLM implementations")
    parser.add_argument("--provider", "-p", choices=["openai", "anthropic"], default="anthropic",  help="Choose VLM provider")
    parser.add_argument("--input", "-i", choices=["text", "image", "audio"], default="text",  help="Choose input type")
    parser.add_argument("--visualize", "-v", default=False, help="Visualize the data")
    args = parser.parse_args()

    #path to audio files
    audio_dir = "/home/mark/github/umi-vlm/data"
    #load audio data from dir
    audio_data, sample_rate, reference_audio_data, test_audio_data = load_audio_files(audio_dir)

    #print dimension of audio data
    # print(f"len: {len(audio_data)}, audio_data[0].shape: {audio_data[0].shape}")
    # sys.exit(0)
    
    #extract audio features from audio data depending on argument input {text, image, audio}
    if args.input == "text":
        audio_features = extract_audio_feature(audio_data, sample_rate)

    elif args.input == "image":
        spectrograms_reference = extract_spectrogram(reference_audio_data, sample_rate)
        spectrograms_test = extract_spectrogram(test_audio_data, sample_rate)

        # Get base64 encoded images for VLM
        ref_image, test_images = visualize_reference_test_spectrograms(
            spectrograms_reference, 
            spectrograms_test, 
            sample_rate,
            plot_spectrograms=False
        )

        # Combine all images for VLM
        image_inputs = [ref_image] + test_images


    elif args.input == "audio":
        pass
    else:
        raise ValueError(f"Invalid input type: {args.input}")

    if args.visualize:
        # For visualizing audio features {text, image, audio}
        if args.input == "text":
            print(json.dumps(audio_features, indent=4))

        elif args.input == "audio":
            pass
    
    # System prompt setting the context
    system_prompt = "You are a helpful assistant with expertise in recognizing patterns and identifying classes based on visual representations of audio data."

    # Combine the prompts with image placeholders
    combined_prompt = '''
Your task is to analyze spectrograms, which are visual representations of the frequency spectrum of sound over time, and to determine the most likely sound class for a given spectrogram.
I will show you several spectrogram visualizations:

First image shows reference spectrograms for three classes: Plastic (left), Fabric (middle), Paper (right).

The following 12 images are individual test spectrograms that need to be classified. For each test spectrogram, analyze it considering factors such as frequency patterns, intensity, and time variations. Focus solely on the patterns presented in the spectrogram.

Please provide your classification for ALL 12 test spectrograms in the following format:

Test 1: [class]
Test 2: [class]
Test 3: [class]
...and so on until Test 12

For each classification, use ONLY one of these exact class names: Plastic, Fabric, or Paper.
'''


    try:
        # Initialize VLM based on provider
        if args.provider == "anthropic":
            vlm = AnthropicVLM(model="claude-3-5-sonnet-20241022")
        else:
            vlm = OpenAIVLM(model="gpt-4o")

        # Run analysis
        print(f"\nUsing provider: {args.provider}")
        print("\nAnalyzing images...")


        # Print dimension of each image
        for i, image in enumerate(image_inputs):
            # Decode base64 string to bytes
            img_bytes = base64.b64decode(image['base64'])
            
            # Convert bytes to numpy array using cv2 or PIL
            import cv2
            import numpy as np
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Print dimensions (height, width, channels)
            print(f"image_{i} dimension: {img.shape}")
        
        
        
        # ***** Send all images and combined prompt to VLM *****
        response = vlm.analyze(
            prompt=combined_prompt,
            images=image_inputs,
            system_prompt=system_prompt
        )
        # ******************************************************

        print("\nResponse:")
        print(response.text)

    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
