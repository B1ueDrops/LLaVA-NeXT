
from deepspeed.profiling.flops_profiler import FlopsProfiler
from deepspeed import init_inference
import torch
from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from ogutils.video import VideoUtils
from ogutils.log import Logger

import copy
import warnings
from PIL import Image
import numpy as np

class InferSession:

    def __init__(
            self,
            model_name='llava_qwen',
            pretrained_path='/root/autodl-tmp/models/llava-onevision-qwen2-7b-ov',
            mm_vision_tower='/root/autodl-tmp/models/siglip-so400m-patch14-384',
            device='cuda:0',
            video_path='/root/autodl-tmp/VSI-Bench/scannet/scene0426_00.mp4',
            prompt='Describe this video and describe the camera motion direction',
            only_key_frames = False,
            dump_attentions = False,
        ):
        self.model_name = model_name
        self.pretrained = pretrained_path
        self.mm_vision_tower = mm_vision_tower
        self.device = device
        self.video_path = video_path
        self.prompt = prompt
        self.only_key_frames = only_key_frames
        self.dump_attentions = dump_attentions

    def run(self):    
        warnings.filterwarnings("ignore")
        if self.device == 'cuda':
            device_map = "auto"
        else:
            device_map = {"": self.device}
        overwrite_config = {}
        overwrite_config["mm_vision_tower"] = self.mm_vision_tower
        llava_model_args = {
            "multimodal": True,
        }
        llava_model_args["overwrite_config"] = overwrite_config
        # tokenzier用下面这段代码产生:
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 其中model_path是/data/lja/models/llava-onevision-qwen2-0.5b-ov
        # model用下面这段代码产生:
        # model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        # image_processor = vision_tower.image_processor
        tokenizer, model, image_processor, max_length = load_pretrained_model(self.pretrained, None, self.model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)
        model.eval()
        video_frames = VideoUtils.load_video(self.video_path)

        # ===
        # img_paths = [
        #     '/root/super_demo/o1.jpg',
        #     '/root/super_demo/w1.png',
        #     '/root/super_demo/o2.jpg',
        #     '/root/super_demo/w2.png',
        #     '/root/super_demo/o3.jpg',
        #     '/root/super_demo/w4.png',
        #     '/root/super_demo/o4.jpg',
        #     '/root/super_demo/w3.png',
        # ]
        # img_arrs = []
        # for img_path in img_paths:
        #     img_arr = np.array(Image.open(img_path))
        #     img_arrs.append(img_arr)
        # video_frames = np.stack(img_arrs, axis=0)
        # ===

        infer_frames = video_frames
        Logger.debug(f'Input Video Tensor Shape: {infer_frames.shape}')

        image_tensors = []
        Logger.info('Processing Frame Tokens...')
        # Frames是图片预处理后的结果
        frames = image_processor.preprocess(infer_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)
        Logger.debug(f'Image Tensor Shape: {image_tensors[0].shape}')
        # Prepare conversation input
        conv_template = "qwen_1_5"
        question = f"{DEFAULT_IMAGE_TOKEN}\n{self.prompt}"

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        # 获取video_token的起始id
        video_token_start_pos = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1].item()
        # 获取所有非video token的总数
        num_text_tokens = input_ids.shape[1]
        image_sizes = [frame.size for frame in infer_frames]
        # Generate response
        prof = FlopsProfiler(model)
        prof.start_profile()
        Logger.info('Start model inference...')
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=["video"],
        )
        # text_output: [batch_size, num_output_tokens]
        # attentions: Tuple(Tuple(torch.Tensor)) [num_output_tokens, num_layers, (batch_size, q_num_heads, seq_len, seq_len)]
        # past_key_values: Tuple(Tuple(torch.Tensor)) [num_layers, 2, (batch_size, kv_head_num, seq_len , head_size)]
        prof.stop_profile()
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        flops = prof.get_total_flops()
        macs  = prof.get_total_macs()
        params = prof.get_total_params()
        duration = prof.get_total_duration()
        #prof.print_model_aggregated_profile()

        # Logger.info(f"Input Text Token Num: {input_ids[0].shape[0]}")
        # Logger.info(f"Output Token Num: {output_tokens.shape[1]}")
        Logger.info(f"FLOPS: {flops}")
        Logger.info(f"MACS: {macs}")
        Logger.info(f"PARAMS: {params}")
        Logger.info(f"DURATIONS: {duration} secs")
        Logger.info(f"Output Text: {text_outputs[0]}")