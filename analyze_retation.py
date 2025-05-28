import numpy as np

file_path1 = '/root/LLaVA-NeXT/vsi_phase1.txt'  # 替换成你的文件名

ratios1 = []
with open(file_path1, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith("Compression Ratio:"):
            value_str = line.split("Compression Ratio:")[1].strip()
            try:
                value = float(value_str)
                ratios1.append(value)
            except ValueError:
                print(f"⚠️ Warning: cannot parse line: {line}")

file_path2 = '/root/LLaVA-NeXT/vsi_phase2.txt'
ratios2 = []
with open(file_path2, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith("Compression Ratio:"):
            value_str = line.split("Compression Ratio:")[1].strip()
            try:
                value = float(value_str)
                ratios2.append(value)
            except ValueError:
                print(f"⚠️ Warning: cannot parse line: {line}")

ratios = []
for i in range(len(ratios1)):
    ratios.append(ratios1[i] * ratios2[i])

# 转成 numpy 数组
ratios = np.array(ratios)

# 计算平均值和方差
mean = np.mean(ratios)
std = np.std(ratios)

print(f"✅ Average (mean): {mean}")
print(f"✅ Standard Deviation (std): {std}")