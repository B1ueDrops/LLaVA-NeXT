import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class DumpUtils:
    @staticmethod
    def visualize_attention(multihead_attention, output_path="atten_map_1.png",title="Layer 5"):
        # Assuming the input is a numpy array of shape (1, num_heads, n_tokens, n_tokens)
        # First, we average the attention scores over the multiple heads
        averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()# Shape: (n_tokens, n_tokens)
        # pooling the attention scores  with stride 20
        averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), 20, stride=20).squeeze(0).squeeze(0)
        cmap = plt.cm.get_cmap("Reds")
        plt.figure(figsize=(7, 5),dpi=400)
        # Log normalization
        log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())
        # set the x and y ticks to 20x of the original
        ax = sns.heatmap(averaged_attention.cpu(),
                    cmap=cmap,  # custom color map
                    norm=log_norm,  # 
                    # cbar_kws={'label': 'Attention score'},
        )
        x_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
        y_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
        ax.set_xticks([i for i in range(0,averaged_attention.shape[0])])
        ax.set_yticks([i for i in range(0,averaged_attention.shape[0])])
        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(y_ticks)
        plt.xticks(fontsize=3)
        plt.yticks(fontsize=3)
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)     
        plt.title(title)
        plt.savefig(output_path, bbox_inches='tight')


    @staticmethod
    def dump_attentions(attentions, output_token_id=0, avg_heads_num=3):
        """
        Used to dump decoding attentions produced by model.generate.
        This attention is got from the process of generating output_token_id th token
        """
        result_path = './result/dump_attentions'
        os.makedirs(result_path, exist_ok=True)
        if output_token_id > len(attentions):
            raise Exception('Output token id exceeds attentions length')
        attentions = attentions[output_token_id]
        num_layers = len(attentions)
        for layer_id in range(num_layers):
            attn_value_each_layer = attentions[layer_id]
            # attn_value_each_layer: (1, q_num_heads, seq_len, seq_len)
            num_heads = attn_value_each_layer.shape[1]
            for head_id in range(0, num_heads, avg_heads_num):
                print(f"[INFO]: Dumping Attention for Layer {layer_id} Head {head_id}...")
                DumpUtils.visualize_attention(
                    attn_value_each_layer[:,head_id:(max(head_id+avg_heads_num, num_heads)),:,:], 
                    output_path=f"{result_path}/attn_layer_{layer_id}_head_{head_id}.png",
                    title=f"Attention Score for Layer {layer_id} Head {head_id}"
                )

    @staticmethod
    def dump_kv_cache(past_key_values):
        result_path = './result/dump_kvcache'
        #os.makedirs(result_path, exist_ok=True)
        pass