from faster_whisper import WhisperModel
import time

model_size = "turbo"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("/home/coder/whisper-trtllm-rs/audio/oppo-en-30s.wav", language="en", beam_size=1, temperature=0.0)

start = time.time()
segments, info = model.transcribe("/home/coder/whisper-trtllm-rs/audio/oppo-en-30s.wav", language="en", beam_size=1, temperature=0.0)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

print("Time taken: ", time.time() - start)