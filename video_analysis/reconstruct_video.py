import cv2
import os
from glob import glob

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
fourcc = cv2.videowriter_fourcc(*'mp4v')
out = cv2.videowriter(output_path, fourcc, fps, (width, height))

# 写入每一帧
for file in frame_files:
    frame = cv2.imread(file)
    out.write(frame)

out.release()