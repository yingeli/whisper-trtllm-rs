INFERENCE_PRECISION=float16
WEIGHT_ONLY_PRECISION=int8
MAX_BEAM_WIDTH=4
MAX_BATCH_SIZE=8
checkpoint_dir=whisper_turbo_weights_${WEIGHT_ONLY_PRECISION}
output_dir=whisper_turbo_${WEIGHT_ONLY_PRECISION}

# Convert the large-v3 turbo model weights into TensorRT-LLM format.
python3 convert_checkpoint.py \
    --use_weight_only \
    --weight_only_precision $WEIGHT_ONLY_PRECISION \
    --output_dir $checkpoint_dir