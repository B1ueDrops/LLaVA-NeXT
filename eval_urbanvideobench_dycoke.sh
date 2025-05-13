#! /bin/bash

#bash /root/LLaVA-NeXT/eval_urbanvideobench_default.sh

export MODEL_PATH='/root/autodl-tmp/models/llava-onevision-qwen2-0.5b-ov'

rm -rf /root/autodl-tmp/UrbanVideo-Bench/urbanvideo-bench
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=8 \
    --main_process_port=25001 \
    -m lmms_eval \
    --model llava_onevision \
   --model_args pretrained=${MODEL_PATH},conv_template=qwen_1_5,model_name=llava_qwen,device_map=auto,dycoke=True \
    --tasks urbanvideobench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path ./urban_logs_dycoke_05b/

export MODEL_PATH='/root/autodl-tmp/models/llava-onevision-qwen2-7b-ov'

rm -rf /root/autodl-tmp/UrbanVideo-Bench/urbanvideo-bench
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=8 \
    --main_process_port=25001 \
    -m lmms_eval \
    --model llava_onevision \
   --model_args pretrained=${MODEL_PATH},conv_template=qwen_1_5,model_name=llava_qwen,device_map=auto,dycoke=True \
    --tasks urbanvideobench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path ./urban_logs_dycoke_7b/
