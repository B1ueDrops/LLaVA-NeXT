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

def sample_32_uniform(items):
    n = len(items)
    if n <= 64:
        return items
    else:
        indices = [int(i * n / 64) for i in range(64)]
        return [items[i] for i in indices]

def calculate_overlap_ratio(corners1, corners2, width, height):
    """
    Calculate the overlap ratio of two polygons (image regions)

    Parameters：
    - corners1: The corner coordinates of the first polygon
    - corners2: The corner coordinates of the second polygon
    - width, height: Width and height of the image

    Return：
    - overlap_ratio: The ratio of overlapping area to total image area
    """
    # Convert coordinates into a format suitable for calculation
    poly1 = np.array([c[0] for c in corners1], dtype=np.float32)
    poly2 = np.array([c[0] for c in corners2], dtype=np.float32)

    # Convert polygon coordinates to a format suitable for cv2.fillPoly
    poly1_int = np.int32([poly1])
    poly2_int = np.int32([poly2])

    # Create black and white images
    img1 = np.zeros((height, width), dtype=np.uint8)
    img2 = np.zeros((height, width), dtype=np.uint8)

    # draw a polygon
    cv2.fillPoly(img1, poly1_int, 1)
    cv2.fillPoly(img2, poly2_int, 1)

    # Calculate overlapping areas
    intersection = cv2.bitwise_and(img1, img2)
    overlap_area = np.sum(intersection)
    total_area = width * height
    overlap_ratio = overlap_area / total_area
    return overlap_ratio

def extract_keyframes(frames, x):
    """
    Extract keyframes from the frame list based on the coincidence ratio.

    Parameters：
    - frames: A list containing video frames, with each frame being a numpy array
    - x: Overlap ratio threshold, between 0 and 1

    Return：
    - keyframe_indices: Index list of keyframes
    """
    keyframe_indices = [0]  # Keep the first frame
    prev_frame = frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

    # Initialize feature detectors and matchers
    orb = cv2.ORB_create()
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for t in range(1, len(frames)):
        curr_frame = frames[t]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        curr_kp, curr_des = orb.detectAndCompute(curr_gray, None)

        # Check if the descriptor is empty
        if prev_des is None or curr_des is None:
            # Unable to match, keep the current frame
            keyframe_indices.append(t)
            prev_frame = curr_frame
            prev_gray = curr_gray
            prev_kp = curr_kp
            prev_des = curr_des
            continue

        # feature matching
        matches = bf.match(prev_des, curr_des)
        matches = sorted(matches, key=lambda x: x.distance)

        # When there are enough matching points, estimate the transformation
        if len(matches) > 10:
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate transformation matrix (e.g. perspective transformation)
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            if M is not None:
                # Calculate the proportion of overlapping areas
                h, w = prev_gray.shape
                corners_prev = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                corners_curr_in_prev = cv2.perspectiveTransform(corners_prev, M)

                # Calculate the overlapping area of two polygons
                overlap_ratio = calculate_overlap_ratio(corners_prev, corners_curr_in_prev, w, h)

                # Determine whether to retain the current frame based on the overlap ratio
                if overlap_ratio < x:
                    keyframe_indices.append(t)
                    # Update the information of the former frame
                    prev_frame = curr_frame
                    prev_gray = curr_gray
                    prev_kp = curr_kp
                    prev_des = curr_des
            else:
                # Unable to calculate transformation, keep current frame
                keyframe_indices.append(t)

                # Update the information of the former frame
                prev_frame = curr_frame
                prev_gray = curr_gray
                prev_kp = curr_kp
                prev_des = curr_des
        else:
            # Too few matching points, keep the current frame
            keyframe_indices.append(t)

            # Update the information of the previous frame
            prev_frame = curr_frame
            prev_gray = curr_gray
            prev_kp = curr_kp
            prev_des = curr_des

    if keyframe_indices[-1] != len(frames) - 1:
        keyframe_indices.append(len(frames) - 1)
    return keyframe_indices

# Load model
# parquet_path = '/root/autodl-tmp/VSI-Bench/test-00000-of-00001.parquet'
# df = pd.read_parquet(parquet_path)
# results = []

# for index, row in df.iterrows():
#     result = {}
#     video_info = row.to_dict()
#     idx = video_info['id']
#     print(f'Processing {idx}/{len(df)}...')
#     dataset = video_info['dataset']
#     scene_name = video_info['scene_name']
#     video_path = f'/root/autodl-tmp/VSI-Bench/{dataset}/{scene_name}.mp4'
#     vr = VideoReader(video_path, ctx=cpu(0))
#     total_frames = len(vr)
#     fps = vr.get_avg_fps()
#     frame_ids = list(range(0, total_frames, int(fps)))
#     frames = vr.get_batch(frame_ids)
#     frames_np = frames.asnumpy()         # 转为 NumPy array，shape 不变
#     frame_list = [frame for frame in frames_np]

#     result['video_id'] = f'{scene_name}.mp4'
#     result['video_path'] = video_path
#     haha = extract_keyframes(frame_list, 0.5)
#     result['er_frames_before'] = haha
#     result['er_frames_after'] = sample_32_uniform(haha)
#     results.append(result)

with open('/root/LLaVA-NeXT/vsi_bench_er_preprocess_haha.json') as f:
    video_infos = json.load(f)
    for i in range(len(video_infos)):
        video_infos[i]['er_frames_after'] = sample_32_uniform(video_infos[i]['er_frames_before'])

    with open("./vsi_bench_er_preprocess_64.json", "w") as f:
        json.dump(video_infos, f, indent=4)
