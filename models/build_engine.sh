# install requirements first

pip install -r requirements.txt

INFERENCE_PRECISION=float16
MAX_BEAM_WIDTH=5
MAX_BATCH_SIZE=8
checkpoint_dir=whisper_turbo_weights
output_dir=whisper_turbo

# Convert the large-v3 turbo model weights into TensorRT-LLM format.
python3 convert_checkpoint.py \
                --model_name large-v3-turbo \
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
              --kv_cache_type continuous \
              --remove_input_padding disable

trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${output_dir}/decoder \
              --moe_plugin disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len 448 \
              --max_input_len 226 \
              --max_encoder_input_len 3000 \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --gpt_attention_plugin ${INFERENCE_PRECISION} \
              --context_fmha disable \
              --kv_cache_type continuous \
              --remove_input_padding disable


python3 run.py --name single_wav_test --engine_dir $output_dir --input_file assets/1221-135766-0002.wav --enable_warmup --use_py_session

python3 run.py --name single_wav_test --engine_dir $output_dir --input_file assets/meeting-30s.wav --use_py_session --enable_warmup

python3 run.py --name single_wav_test --engine_dir $output_dir --input_file ../audio/oppo-en-30s.wav --enable_warmup --use_py_session

trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${output_dir}/decoder \
              --moe_plugin disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len 1000 \
              --max_input_len 256 \
              --max_encoder_input_len 3000 \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --context_fmha disable


INFERENCE_PRECISION=float16
MAX_BEAM_WIDTH=5
MAX_BATCH_SIZE=8
checkpoint_dir=whisper_turbo_weights
output_dir=whisper_turbo

python3 convert_checkpoint.py \
              --model_name large-v3-turbo \
              --dtype ${INFERENCE_PRECISION} \
              --logits_dtype ${INFERENCE_PRECISION} \
              --output_dir $checkpoint_dir

trtllm-build  --checkpoint_dir ${checkpoint_dir}/encoder \
              --output_dir ${output_dir}/encoder \
              --moe_plugin disable \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --gemm_plugin disable \
              --max_input_len 3000 --max_seq_len=3000

trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${output_dir}/decoder \
              --moe_plugin disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len 448 \
              --max_input_len 224 \
              --max_encoder_input_len 3000 \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --gpt_attention_plugin ${INFERENCE_PRECISION}