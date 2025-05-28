import numpy as np
from decord import VideoReader, cpu

class VideoUtils:
    @staticmethod
    def load_video(video_path, fps: int=1):
        """
        Load video, sampling 'fps' frames per second.
        Returns:
            sampled_frames: np.ndarray of shape (num_frames, height, width, channels)
        """
        if isinstance(video_path, str):
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))

        total_frame_num = len(vr)
        video_fps = vr.get_avg_fps()
        video_duration = total_frame_num / video_fps  # 秒

        frame_idx = []

        for sec in range(int(video_duration)):
            # 计算当前秒在原始帧中的范围
            start_frame = int(sec * video_fps)
            end_frame = int((sec + 1) * video_fps)
            if end_frame > total_frame_num:
                end_frame = total_frame_num

            # 在这一秒内均匀采样 fps 个点
            if end_frame > start_frame:
                samples = np.linspace(start_frame, end_frame - 1, fps, dtype=int)
                frame_idx.extend(samples.tolist())

        frame_idx = np.clip(frame_idx, 0, total_frame_num - 1)  # 避免越界

        sampled_frames = vr.get_batch(frame_idx).asnumpy()
        return sampled_frames  # (num_frames, height, width, channels)
        
    @staticmethod
    def load_video_with_keyframes(video_path):
        """
        Load Video with all frame nums and keyframe index
        """
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, total_frame_num, dtype=int)
        all_frame_idx = uniform_sampled_frames.tolist()
        key_frame_idx = vr.get_key_indices()
        spare_frames = vr.get_batch(all_frame_idx).asnumpy()
        key_frames = vr.get_batch(key_frame_idx).asnumpy()
        return spare_frames, key_frames, key_frame_idx  # (frames, height, width, channels)