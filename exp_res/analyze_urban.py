import numpy as np

def parse_to_list(file_path, keyword):
    res = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{keyword}:"):
                value_str = line.split(f"{keyword}:")[1].strip()
                try:
                    value = float(value_str)
                    res.append(value)
                except ValueError:
                    print(f"⚠️ Warning: cannot parse line: {line}")
    return np.array(res)

def print_avg_std(arr_np):
    mean = np.mean(arr_np)
    std = np.std(arr_np)

    return f"{mean:.2f} ± {std:.2f}"

tmp1 = parse_to_list('./urban_er_with_ours.txt', 'Compression Ratio')
tmp2 = parse_to_list('./urban_homo2.txt', 'Compression Ratio')
ours_comp_ratio = np.array([tmp1[i] * tmp2[i] for i in range(min(len(tmp1), len(tmp2)))])
ours_flops = parse_to_list('./urban_ours_with_er_flops.txt', 'flops')
ours_flops = ours_flops / 1e12
ours_prefill = parse_to_list('./urban_ours_with_er_prefill.txt', 'Prefill')
ours_decode = parse_to_list('./urban_ours_with_er_decode.txt', 'Decode')
ours_kvcache = parse_to_list('./urban_ours_with_er_kvcache.txt', 'kvcache')

print('Ours:')
print(f'Comp Ratio: {print_avg_std(ours_comp_ratio)}')
print(f'TFLOPS: {print_avg_std(ours_flops)}')
print(f'Prefill Time(s): {print_avg_std(ours_prefill)}')
print(f'Decode Time (s): {print_avg_std(ours_decode)}')
print(f'KV Cache Size (MB): {print_avg_std(ours_kvcache)}')

er_comp_ratio = tmp1
er_flops = parse_to_list('./urban_er_without_ours_flops.txt', 'flops')
er_flops = er_flops / 1e12
er_prefill = parse_to_list('./urban_er_without_ours_prefill.txt', 'Prefill')
er_decode = parse_to_list('./urban_er_without_ours_decode.txt', 'Decode')
er_kvcache = parse_to_list('./urban_er_without_ours_kvcache.txt', 'kvcache')

print()
print('ER:')
print(f'Comp Ratio: {print_avg_std(er_comp_ratio)}')
print(f'TFLOPS: {print_avg_std(er_flops)}')
print(f'Prefill Time(s): {print_avg_std(er_prefill)}')
print(f'Decode Time (s): {print_avg_std(er_decode)}')
print(f'KV Cache Size (MB): {print_avg_std(er_kvcache)}')

static_comp_ratio = parse_to_list('./urban_homo1.txt', 'Compression Ratio')
static_flops = parse_to_list('./urban_static_flops.txt', 'flops')
static_flops = static_flops / 1e12
static_prefill = parse_to_list('./urban_static_prefill.txt', 'Prefill')
static_decode = parse_to_list('./urban_static_decode.txt', 'Decode')
static_kvcache = parse_to_list('./urban_static_kvcache.txt', 'kvcache')

print()
print('Static:')
print(f'Comp Ratio: {print_avg_std(static_comp_ratio)}')
print(f'TFLOPS: {print_avg_std(static_flops)}')
print(f'Prefill Time(s): {print_avg_std(static_prefill)}')
print(f'Decode Time (s): {print_avg_std(static_decode)}')
print(f'KV Cache Size (MB): {print_avg_std(static_kvcache)}')

full_flops = parse_to_list('./urban_full_flops.txt', 'flops')
full_flops = full_flops / 1e12
full_prefill = parse_to_list('./urban_full_prefill.txt', 'Prefill')
full_decode = parse_to_list('./urban_full_decode.txt', 'Decode')
full_kvcache = parse_to_list('./urban_full_kvcache.txt', 'kvcache')

print()
print('Full:')
print(f'TFLOPS: {print_avg_std(full_flops)}')
print(f'Prefill Time(s): {print_avg_std(full_prefill)}')
print(f'Decode Time (s): {print_avg_std(full_decode)}')
print(f'KV Cache Size (MB): {print_avg_std(full_kvcache)}')