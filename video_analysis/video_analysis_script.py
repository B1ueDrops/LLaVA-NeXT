import os
import json
from decord import VideoReader
from decord import cpu
import cv2
from glob import glob
from PIL import Image, ImageDraw, ImageFont

# 设置路径
video_path = '/root/autodl-tmp/VSI-Bench/arkitscenes/41069025.mp4'
output_dir = './frames'

video_name = os.path.basename(video_path)

target_video_info = None
with open('/root/LLaVA-NeXT/preprocess/vsi_bench_motion.json') as m:
    video_infos = json.load(m)
    for video_info in video_infos:
        if video_info['video_id'] == video_name:
            target_video_info = video_info
            break

motion_list = target_video_info['motion_list']
score_list = target_video_info['scores']


# 创建保存帧的目录
os.makedirs(output_dir, exist_ok=True)

# 加载视频
# 加载视频
vr = VideoReader(video_path, ctx=cpu(0))
fps = vr.get_avg_fps()  # 获取视频帧率

# 每秒采样一帧
interval = int(fps)  # 每隔 fps 帧采一帧

cnt = 0
for i in range(0, len(vr), interval):
    frame = vr[i]
    # 获取 motion 文本
    if int(i / interval) < len(motion_list):
        motion = motion_list[int(i / interval)]
        score = score_list[int(i / interval)]
    else:
        motion = ""
        score = ""
    # 转换为 PIL 图像
    img = Image.fromarray(frame.asnumpy())

    # 创建可画图的对象
    draw = ImageDraw.Draw(img)

    # 设置字体（可选：指定字体文件，否则使用默认字体）
    try:
        font = ImageFont.truetype("arial.ttf", 24)  # Windows 通常自带 arial
    except:
        font = ImageFont.load_default()

    # 写字：位置 (10,10)，白色，带黑边
    draw.text((10, 10), motion, fill="white", font=font, stroke_width=2, stroke_fill="black")
    draw.text((10, 30), str(score), fill="white", font=font, stroke_width=2, stroke_fill="black")

    # 保存图片
    img.save(os.path.join(output_dir, f'{i:05d}.jpg'))
    cnt += 1


# 参数配置
frame_dir = './frames'
output_path = 'new_video.mp4'
fps = 1  # 每秒1帧

# 读取所有帧文件，并排序
frame_files = sorted(glob(os.path.join(frame_dir, '*.jpg')))

# 读取第一帧确定尺寸
frame = cv2.imread(frame_files[0])
height, width, _ = frame.shape

# 定义视频编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 写入每一帧
for file in frame_files:
    frame = cv2.imread(file)
    out.write(frame)

out.release()