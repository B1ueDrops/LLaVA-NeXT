#! /bin/bash
export MODEL_PATH='/root/autodl-tmp/models/llava-onevision-qwen2-0.5b-ov'
accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=${MODEL_PATH},conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen \
    --tasks=vsibench \
    --batch_size=1 \
    --verbosity=DEBUG