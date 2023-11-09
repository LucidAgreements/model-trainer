import torch
import whisper

# OPTION 2: Load the Whisper model:
# model = whisper.load_model('/Users/aliuspetraska/Documents/Git/model-trainer/models/test_1.pt')

# OPTION 1: Load the Whisper model:
model = whisper.load_model("medium").load_state_dict(torch.load('./models/test_4.pt'))

# Let's get the transcript of the temporary file
result = model.transcribe("./input/1-video.wav")

print(result["text"])