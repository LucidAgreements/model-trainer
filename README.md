## Setup

```bash
pip3 install -r ./requirements.txt
```

It also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

## Build

User Jupyter Notebook to train and store model.

## Convert the model

1. Place Your Hugging Face Model: Ensure your Hugging Face model file (e.g., pytorch_model.bin) is saved in a known directory.

2. Run the Script: Use the command line to navigate to the directory containing the script and run it with the necessary arguments.

3. The script requires three arguments:
--hf_model_path: Path to the Hugging Face model file.
--whisper_model_path: Path where the converted Whisper model should be saved.
--model: Size of the Whisper model (options are 'tiny', 'base', 'small', 'medium', 'large').

Here's an example command:
```bash
python hf_to_openai_converter.py --hf_model_path "path/to/hf_model.bin" --whisper_model_path "path/to/whisper_model.pt" --model "medium"
```

## Test

```bash
python test.py --model "medium" --custom_model_path "path/to/whisper_model.pt" --audio_path "path/to/audio.wav"
```

## Research Hints & Errors

![image_1](images/image_1.png)

![image_2](images/image_2.png)

![image_3](images/image_3.png)

![image_4](images/image_4.png)
