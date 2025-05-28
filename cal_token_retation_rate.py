
from PIL import Image
import requests
import copy
import torch
import time
import cv2
import numpy as np
import os

import sys
import json
import warnings
from tqdm import tqdm
import pandas as pd
from decord import VideoReader
from decord import cpu  # 默认使用 CPU
from tqdm import tqdm

# Load model
# parquet_path = '/root/autodl-tmp/VSI-Bench/test-00000-of-00001.parquet'
# df = pd.read_parquet(parquet_path)
# results = []

# cnt = 0
# for index, row in df.iterrows():
#     result = {}
#     video_info = row.to_dict()
#     dataset = video_info['dataset']
#     scene_name = video_info['scene_name']
#     video_path = f'/root/autodl-tmp/VSI-Bench/{dataset}/{scene_name}.mp4'
#     vr = VideoReader(video_path, ctx=cpu(0))
#     total_frames = len(vr)
#     fps = vr.get_avg_fps()
#     frame_ids = list(range(0, total_frames, int(fps)))
#     cnt += len(frame_ids)
# print(cnt)

cnt = 0
with open('/root/LLaVA-NeXT/vsi_bench_er_preprocess.json') as f:
    video_infos = json.load(f)
    for video_info in video_infos:
        cnt += len(video_info['er_frames_after'])
print(cnt)