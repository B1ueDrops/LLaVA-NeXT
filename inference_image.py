from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from deepspeed.profiling.flops_profiler import FlopsProfiler

from PIL import Image
import requests
import copy
import torch

import sys
import warnings

# Load model
pretrained = "/root/autodl-tmp/models/llava-onevision-qwen2-0.5b-ov"
model_name = "llava_qwen"
device="cuda"
device_map = {"": "cuda:0"}
llava_model_args = {
    "multimodal": True,
}
overwrite_config = {}
overwrite_config["image_aspect_ratio"] = "pad"
overwrite_config["mm_vision_tower"] = "/root/autodl-tmp/models/siglip-so400m-patch14-384"
llava_model_args["overwrite_config"] = overwrite_config
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation='sdpa', **llava_model_args)

model.eval()

# Load two images
url1 = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
url2 = "https://raw.githubusercontent.com/haotian-liu/LLaVA/main/images/llava_logo.png"

# image1 = Image.open(requests.get(url1, stream=True).raw)
# image2 = Image.open(requests.get(url2, stream=True).raw)
image1 = Image.open('./egocache/frame2.png')
image2 = Image.open('./egocache/warped_frame.png')

images = [image2]
image_tensors = process_images(images, image_processor, model.config)
image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]

# Prepare interleaved text-image input
conv_template = "qwen_1_5"
question = f"{DEFAULT_IMAGE_TOKEN} This is the first image. Can you describe what you see?\n\nNow, let's look at another image: {DEFAULT_IMAGE_TOKEN}\nWhat's the difference between these two images?"

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size for image in images]

prof = FlopsProfiler(model)
prof.start_profile()
# Generate response
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
prof.stop_profile()
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])
flops = prof.get_total_flops()
macs  = prof.get_total_macs()
params = prof.get_total_params()
duration = prof.get_total_duration()
#prof.print_model_aggregated_profile()

# Logger.info(f"Input Text Token Num: {input_ids[0].shape[0]}")
# Logger.info(f"Output Token Num: {output_tokens.shape[1]}")
print(f"FLOPS: {flops}")
print(f"MACS: {macs}")
print(f"PARAMS: {params}")
print(f"DURATIONS: {duration} secs")
print(f"Output Text: {text_outputs[0]}")