
# ViT处理流程
- 假设有n帧, 首先将图片预处理成`(n, 3, 384, 384)`的张量.
- 然后, patch大小是14 * 14
- 一帧中有27 * 27个token
- 然后对于每一帧, 通过`mm_patch_merge_type = 'spatial_unpad'`, 将每一帧处理成一个patch
- 最终一帧的token个数是14 * 14 * n
- 源码文件在: `llava/model/language_model/llava_qwen.py`