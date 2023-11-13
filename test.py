import torch
import whisper
import argparse

def transcribe(model :str,
               audio_path :str,
               custom_model_path :str,
               device = "cpu"):
    # Load a standard Whisper model
    model = whisper.load_model(model, device=device)
    # Load custom model weights
    model.load_state_dict(torch.load(custom_model_path, map_location=torch.device(device)))
    # Transcribe audio file
    result = model.transcribe(audio=audio_path)
    print(result["text"])

def main():
     # Create argument parser
    parser = argparse.ArgumentParser(description='Transcribe audio using a Whisper model.')
    parser.add_argument('--model',
                        type=str,
                        default='medium',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Size of the Whisper model.')
    parser.add_argument('--custom_model_path',
                        type=str,
                        help='Path to custom Whisper model weights (optional).')
    parser.add_argument('--audio_path',
                        type=str,
                        required=True,
                        help='Path to the audio file to transcribe.')

    # Parse arguments
    args = parser.parse_args()
    # Transcribe audio file
    transcribe(model=args.model,
               audio_path=args.audio_path,
               custom_model_path=args.custom_model_path)

if __name__ == "__main__":
    main()