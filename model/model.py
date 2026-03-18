from transformers import PretrainedConfig
import torch
import torch.nn as nn
from typing import Optional
import math

# huggingface 的类 (这里不用管 直接复制即可)
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int | None = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )




# 继承nn.Module类
class RMSNorm(nn.Module):
    # __init__初始化
    # dim: 每个token的向量长度 常见输入为[batch, sequence_len, dim] dim = d_model
    # eps: epsilon大小
    def __init__(self, dim:int, eps:float = 1e-5):
        super().__init__()
        
        self.dim = dim
        self.eps = eps
        # 1*dim
        self.weight = nn.Parameter(torch.ones(dim))

    # _norm
    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps) 
    
    # nn.Module要求必须写forward
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x) * x



# end: 推断的长度
# rope_base: 常取10000
# rope_scaling: 缩放方法，Optional默认None, 也可传入dict 等价于dict | None
def precompute_freqs_cis(dim:int, end:int = 32*1024, rope_base:int = 10000, rope_scaling:Optional[dict] = None):
    # 初始化RoPE频率
    # [:(dim//2)]保证数组长度为 dim//2 这里dim一般取偶数 有些冗余
    # freqs = theta 
    # atnn_factor 温度缩放
    freqs, attn_factor = (1.0 / (rope_base ** (torch.arange(0, dim, 2)[:(dim//2)].float()/dim)), 1.0)

    # 有缩放方法时
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"], # 缩放因子
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"],
        )

    # 推断的长度大于训练长度，使用缩放
    if end > orig_max:
        # 波长b到i的映射 输入b的值通过计算得到inv_dim
        inv_dim = lambda b:(dim*math.log(orig_max/(b*2*math.pi)))/(2*math.log(rope_base))
        
        # 划分高低维度
        # low低维: 不许要缩放的高频部分
        # high高维: 需要缩放的低频部分
        low, high = (max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim//2 - 1))

        # 计算缩放因子
        # low之前, ramp为0, 在high之后ramp为1, 在low和high之间线性变化
        ramp = torch.clamp((torch.arange(dim//2, device = freqs.device).float()-low)/max(high-low, 1), 0, 1)

        # 当ramp为0时, freqs不变; 当ramp为1时, freqs缩放为freqs*factor; 在0和1之间线性变化
        freqs = freqs*(1- ramp + ramp/factor)

    # 根据end生成位置索引t
    # t = i
    t = torch.arange(end, device = freqs.device).float()

    # 计算外积，将t和频率部分相乘，得到每个位置的旋转角度
    freqs = torch.outer(t, freqs).float()

    # 将维度补回到dim 每两个是一组要降维然后再补全
    freqs_cos = (
        torch.cat([torch.cos(freqs), torch.cos(freqs)], dim = -1) * attn_factor
    )
    freqs_sin = (
        torch.cat([torch.sin(freqs), torch.sin(freqs)], dim = -1) * attn_factor
    )
    return freqs_cos, freqs_sin

# 编写 RoPE
def apply_rotary_pos_emb(q, k, cos, sin, position_ids = None, unsqueeze_dim = 1):
    # [a, b] -> [-b, a]
    def rotate_half(x):
        return torch.cat(
            # 取最后一维的后半部分取负 取最后一维的前半部分不变 然后拼接起来
            (-x[... , x.shape[-1]//2 :], x[... , :x.shape[-1]//2]), dim = -1
        )

    # TODO: 这里没有采用两两通道交错成对旋转 后续考虑两种方法差异
    # 只要固定且一致的配对即可 不需要交错成对
    # x_rotated = x*cos + rotated_half(x)*sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

