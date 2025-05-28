#! /bin/bash

rm -rf /root/autodl-tmp/UrbanVideo-Bench/urban_video-bench
export MODEL_PATH='/root/autodl-tmp/models/llava-onevision-qwen2-7b-ov'
accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=${MODEL_PATH},conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen,enable_er=True,homo_type=2,task_nam=urban \
    --tasks=urbanvideobench \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path ./exp_res/exp_urban_ours_logs/

rm -rf /root/autodl-tmp/UrbanVideo-Bench/urban_video-bench
export MODEL_PATH='/root/autodl-tmp/models/llava-onevision-qwen2-7b-ov'
accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=${MODEL_PATH},conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen,enable_er=True,homo_type=0,task_nam=urban \
    --tasks=urbanvideobench \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path ./exp_res/exp_urban_er_logs/


rm -rf /root/autodl-tmp/UrbanVideo-Bench/urban_video-bench
export MODEL_PATH='/root/autodl-tmp/models/llava-onevision-qwen2-7b-ov'
accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=${MODEL_PATH},conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen,enable_er=False,homo_type=1,task_nam=urban \
    --tasks=urbanvideobench \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path ./exp_res/exp_urban_static_logs/

rm -rf /root/autodl-tmp/UrbanVideo-Bench/urban_video-bench
export MODEL_PATH='/root/autodl-tmp/models/llava-onevision-qwen2-7b-ov'
accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=${MODEL_PATH},conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen,enable_er=False,homo_type=0,task_nam=urban \
    --tasks=urbanvideobench \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path ./exp_res/exp_urban_full_logs/
