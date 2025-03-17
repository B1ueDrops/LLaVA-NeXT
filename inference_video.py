from deepspeed.profiling.flops_profiler import FlopsProfiler
from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from ogutils.dump import dump_attentions

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = "/data/lja/models/llava-onevision-qwen2-0.5b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = {"": "cuda:0"}
overwrite_config = {}
overwrite_config["mm_vision_tower"] = "/data/lja/models/siglip-so400m-patch14-384"
llava_model_args = {
    "multimodal": True,
}
llava_model_args["overwrite_config"] = overwrite_config
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)

model.eval()


# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


# Load and process video
video_path = "/home/lja/static/jobs.mp4"
video_frames = load_video(video_path, 16)
print(video_frames.shape) # (16, 1024, 576, 3)
image_tensors = []
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
image_tensors.append(frames)

# Prepare conversation input
conv_template = "qwen_1_5"
question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [frame.size for frame in video_frames]

# Generate response
prof = FlopsProfiler(model)
prof.start_profile()
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
    output_attentions=True,
    modalities=["video"],
    use_cache=True,
    return_dict_in_generate=True
)
# text_output: [batch_size, num_output_tokens]
# attentions: Tuple(Tuple(torch.Tensor)) [num_output_tokens, num_layers, (batch_size, q_num_heads, seq_len, seq_len)]
# past_key_values: Tuple(Tuple(torch.Tensor)) [num_layers, 2, (batch_size, kv_head_num, seq_len , head_size)]
output_tokens, attentions, past_key_values = cont['sequences'], cont['attentions'], cont['past_key_values']
dump_attentions(attentions)

prof.stop_profile()
text_outputs = tokenizer.batch_decode(output_tokens[0], skip_special_tokens=True)
flops = prof.get_total_flops()
macs  = prof.get_total_macs()
params = prof.get_total_params()
duration = prof.get_total_duration()
prof.print_model_aggregated_profile()
print(f"num output tokens: {output_tokens.shape[1]}")
print(f"flops: {flops}, macs: {macs}, params: {params}, duration: {duration}")
print(f"prompt length: {input_ids[0].shape[0]}")
print(f"output text: {text_outputs[0]}")