import whisper
import time

model = whisper.load_model("large-v2", device="cuda")
result = model.transcribe("/home/coder/whisper-trtllm-rs/audio/whisper.wav", beam_size=5)
print(result)