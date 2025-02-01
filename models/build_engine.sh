# install requirements first


pip install -r requirements.txt

INFERENCE_PRECISION=float16
WEIGHT_ONLY_PRECISION=int8
MAX_BEAM_WIDTH=4
MAX_BATCH_SIZE=8
checkpoint_dir=whisper_turbo_weights_${WEIGHT_ONLY_PRECISION}
output_dir=whisper_turbo_${WEIGHT_ONLY_PRECISION}

# Convert the large-v3 turbo model weights into TensorRT-LLM format.
python3 convert_checkpoint.py \
                --model_name large-v3-turbo \
                --use_weight_only \
                --weight_only_precision $WEIGHT_ONLY_PRECISION \
                --output_dir $checkpoint_dir

# Build the large-v3 model using trtllm-build
trtllm-build  --checkpoint_dir ${checkpoint_dir}/encoder \
              --output_dir ${output_dir}/encoder \
              --moe_plugin disable \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --gemm_plugin disable \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --max_input_len 3000 --max_seq_len=3000 \
              --context_fmha disable \
              --remove_input_padding disable \
              --kv_cache_type continuous


trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${output_dir}/decoder \
              --moe_plugin disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len 114 \
              --max_input_len 14 \
              --max_encoder_input_len 3000 \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --gpt_attention_plugin ${INFERENCE_PRECISION} \
              --context_fmha disable \
              --remove_input_padding disable \
              --kv_cache_type continuous

python3 run.py --name single_wav_test --engine_dir $output_dir --input_file assets/1221-135766-0002.wav --use_py_session

python3 run.py --name single_wav_test --engine_dir $output_dir --input_file assets/meeting-30s.wav --enable_warmup --use_py_session 


trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${output_dir}/decoder \
              --moe_plugin disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len 114 \
              --max_input_len 14 \
              --max_encoder_input_len 3000 \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --context_fmha disable