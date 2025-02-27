import whisper
import time

model = whisper.load_model("turbo")
result = model.transcribe("/home/coder/whisper-trtllm-rs/models/assets/tcl.wav")
print(result["text"])