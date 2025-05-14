import os
import json
import base64
from tqdm import tqdm
from PIL import Image
from openai import OpenAI
from decord import VideoReader, cpu

motion_options = [
    'forward', 'backward', 'left', 'right', 'upward',
    'downward', 'yaw left', 'yaw right', 'pitch up',
    'pitch down', 'roll left', 'roll right'
]

prompt = """
You are a vision system that analyzes egocentric (first-person) camera motion.

You will be given two consecutive video frames captured from a first-person camera.  
Your task is to determine the relative motion of the camera between these two frames.

Only choose **one** of the following 12 motion directions that best describes the motion:
- forward, backward, left, right, upward, downward  
- yaw left, yaw right, pitch up, pitch down, roll left, roll right

Do **not** explain your answer. Do **not** output anything other than the exact motion direction string.

Input:
[Frame 1: <insert first frame or its representation>]  
[Frame 2: <insert second frame or its representation>]

Answer:
"""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_qwen(img1_base, img2_base):
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {
                'role': 'system',
                'content': [
                    {"type": "text", "text": "You are a helpful assistant."}
                ]
            },
            {
                'role': 'user',
                'content': [
                    {
                        'type': "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img1_base}"
                        }
                    },
                    {
                        'type': "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img2_base}"
                        }
                    },
                    {
                        'type': "text", "text": prompt
                    },
                ]
            }
        ]
    )
    return completion.choices[0].message.content

if __name__ == '__main__':
    dataset_path = '/root/LLaVA-NeXT/preprocess/vsi_bench_tmp.json'
    with open('/root/LLaVA-NeXT/preprocess/vsi_duplicate.json') as fp:
        duplicate_dict = json.load(fp)
    with open(dataset_path) as f:
        video_infos = json.load(f)
        for k in tqdm(range(len(video_infos))):
        # for k in range(1):
            print(f'Processing {k}/{len(video_infos)}')
            video_info = video_infos[k]
            motion_list = []
            video_id = video_info['video_id']
            if video_id in duplicate_dict and None not in duplicate_dict[video_id]:
                video_infos[k]['motion_list'] = duplicate_dict[video_id]
                continue
            if 'urbanvideo' in dataset_path:
                question_id = video_info['Question_id']
            else:
                question_id = video_info['question_id']
            video_path = f'/root/autodl-tmp/VSI-Bench/videos/{video_id}'
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame_num = len(vr)
            video_frame_idxs = video_info['frames']
            for i in tqdm(range(len(video_frame_idxs))):
                if i == len(video_frame_idxs) - 1:
                    continue
                idx1 = video_frame_idxs[i]
                idx2 = video_frame_idxs[i + 1]
                img1 = Image.fromarray(vr[idx1].asnumpy())
                img2 = Image.fromarray(vr[idx2].asnumpy())
                img1.save(f'/root/autodl-tmp/tmp/{question_id}_frame_{idx1}.png')
                img2.save(f'/root/autodl-tmp/tmp/{question_id}_frame_{idx2}.png')
                img1_base = encode_image(f'/root/autodl-tmp/tmp/{question_id}_frame_{idx1}.png')
                img2_base = encode_image(f'/root/autodl-tmp/tmp/{question_id}_frame_{idx2}.png')
                try:
                    diff = call_qwen(img1_base, img2_base)
                except Exception as e:
                    img1_path = f"/root/autodl-tmp/tmp/{question_id}_frame_{idx1}.png"
                    if os.path.exists(img1_path):
                        os.remove(img1_path)
                    else:
                        print("img1不存在")
                    img2_path = f"/root/autodl-tmp/tmp/{question_id}_frame_{idx2}.png"
                    if os.path.exists(img2_path):
                        os.remove(img2_path)
                    else:
                        print("img1不存在")
                    with open('/root/LLaVA-NeXT/preprocess/vsi_duplicate.json', 'w') as fff:
                        json.dump(duplicate_dict, fff, indent=4)
                    continue
                motion_list.append(diff)
                img1_path = f"/root/autodl-tmp/tmp/{question_id}_frame_{idx1}.png"
                if os.path.exists(img1_path):
                    os.remove(img1_path)
                else:
                    print("img1不存在")
                img2_path = f"/root/autodl-tmp/tmp/{question_id}_frame_{idx2}.png"
                if os.path.exists(img2_path):
                    os.remove(img2_path)
                else:
                    print("img1不存在")
            video_infos[k]['motion_list'] = motion_list
            duplicate_dict[video_id] = motion_list
        with open('/root/LLaVA-NeXT/preprocess/vsi_bench_motion.json', 'w') as ff:
            json.dump(video_infos, ff, indent=4)
        
        with open('/root/LLaVA-NeXT/preprocess/vsi_duplicate.json', 'w') as fff:
            json.dump(duplicate_dict, fff, indent=4)