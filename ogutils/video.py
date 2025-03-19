import numpy as np
from decord import VideoReader, cpu

class VideoUtils:
    @staticmethod
    def load_video(video_path, max_frames_num):
        """
        Load Video with specific frame nums
        """
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)
    
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