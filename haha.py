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
parquet_path = '/root/autodl-tmp/VSI-Bench/test-00000-of-00001.parquet'
df = pd.read_parquet(parquet_path)
results = []

cnt = 0
with open('/root/LLaVA-NeXT/vsi_bench_er_preprocess.json') as f:
    video_infos_ori = json.load(f)
    for index, row in df.iterrows():
        video_info = row.to_dict()
        video_infos_ori[cnt]['question'] = video_info['question']
        cnt += 1


with open("./vsi_bench_er_preprocess_haha.json", "w") as f:
    json.dump(video_infos_ori, f, indent=4)
