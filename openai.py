import whisper
import time

model = whisper.load_model("turbo")
result = model.transcribe("/home/coder/whisper-trtllm-rs/models/assets/oppo-th-th.wav", beam_size=5, temperature=0.0)
print(result)