import whisper

model = whisper.load_model("turbo")
options = {
    "language": "en",
}
result = model.transcribe("/home/coder/whisper-trtllm-rs/models/assets/oppo-en-us.wav")
print(result)