#! /bin/bash

export MODEL_PATH='/root/autodl-tmp/models/llava-onevision-qwen2-0.5b-ov'

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=8 \
    --main_process_port=25001 \
    -m lmms_eval \
    --model llava_onevision \
   --model_args pretrained=${MODEL_PATH},conv_template=qwen_1_5,model_name=llava_qwen,device_map=auto,dycoke=True \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path ./logs/