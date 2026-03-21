from transformers import PretrainedConfig
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math
from torch.nn import functional as F
from transformers.activations import ACT2FN





# huggingface 的类 (这里不用管 直接复制即可)
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512, # d_model维度 被隐藏
        intermediate_size: int | None = None, # FFN中 升维的维度
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8, # Q的头数量 K和V的头数量由num_key_value_heads控制
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2, # K和V的头数量
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
    if rope_scaling is not None and end > orig_max:
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



# 一个token分成一个QKV, Q获得的头更多一些, K和V获得的头更少一些
# 把较少的 K/V 头复制扩展到和 Q 头数量一致，方便做注意力计算 需要维度复制不只是数字变换
def repeat_kv(x:torch.Tensor, n_rep:int) -> torch.Tensor:
    # batch, seq_len, num_key_value_heads(头的数量), head_dim
    bs, slen, num_key_value_heads, head_dim = x.shape

    # 不需要重复复制 直接返回
    if n_rep == 1:
        return x

    # 复制 n_rep 次
    # [bs, slen, num_key_value_heads, 1, head_dim]
    return (x[:, :, :, None, :]
            # 把第4维从 1 广播成 n_rep -> [1 , 1, 1, ...]
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim))


class Attention(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()

        # 如果传入了num_key_value_heads就用传入的 否则默认和num_attention_heads一样
        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads

        # 要确保Q是K/V头数量的整数倍
        assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads # 当前使用的头的数量
        self.n_kv_heads = self.num_key_value_heads # K和V的头数量
        self.n_rep = self.n_local_heads // self.n_kv_heads # K/V头复制的次数
        self.head_dim = args.hidden_size // args.num_attention_heads # 每个头的维度 

        # 投影层
        # 把最后一维拆成 num_attention_heads * head_dim 或 num_key_value_heads * head_dim 获得 head_dim
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias = False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)

        # 输出 还原维度
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias = False)
        
        # 注意力内部attention dropout
        self.dropout = nn.Dropout(args.dropout)
        # 残差网络
        self.resid_dropout = nn.Dropout(args.dropout)

        # 查看程序是否支持flash_attention 让attention计算更迅速
        # 可以内部实现也可以手动实现
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attention

    
    

    def forward(self, x:torch.Tensor, position_embedding:Tuple[torch.Tensor, torch.Tensor], past_key_value:Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache = False, attention_mask:Optional[torch.Tensor]=None) -> torch.Tensor:
        # attention_mask: [bsz, seq_len] 1表示可以注意 0表示不能注意 用于处理padding部分
        
        # 投影，计算出QKV
        bsz, seq_len, _ = x.size()
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 把输入拆分成多个头 用view
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # 对于Q和K使用RoPE
        cos, sin = position_embedding
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # 对于K和V使用repeat (注意kv cache)
        # 拼接是在 seq_len 维度操作的，而广播是在 heads 维度操作的
        if past_key_value is not None:
            torch.cat([past_key_value[0], xk], dim = 1)
            torch.cat([past_key_value[1], xv], dim = 1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            # 交换1，2维度 变成 [bsz, n_local_heads, seq_len, head_dim] 和 [bsz, n_kv_heads * n_rep, seq_len, head_dim]
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # 进行attention计算，q@k^T / sqrt(head_dim)
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (
                None if attention_mask is None else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()          
            )
            # 训练模式 or 推理模式
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask, dropout_p = self.dropout.p if self.training else 0.0, is_causal = True)
        else:
            # 这个是后两个维度的矩阵乘法，得到每个头的注意力分数 维度为 [bsz, n_local_heads, seq_len, seq_len]
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 注意力掩码
            # 对角线以下变成负无穷 这样经过softmax后就变成0了
            # 维度变为 [1, 1, seq_len, seq_len]才能和scores广播相加
            scores = scores + torch.tril((torch.full((seq_len, seq_len), float("-inf"), device = scores.device)), diagonal = 1).unsqueeze(0).unsqueeze(0)

            if attention_mask is not None:
                # [bsz, 1, 1, seq_len]
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                # padding部分变成负无穷 这样经过softmax后就变成0了
                scores = scores + extended_attention_mask
        
            scores = F.softmax(scores.float(), dim = -1).type_as(scores)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 最后拼接头，输出投影，返回
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    # 初始化
    # 升维
    # 降维
    # 门控
    # dropout
    # 激活函数
    def __init__(self, args:MokioMindConfig):
        super().__init__()
        
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            # 向上取整成64的倍数
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
            
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias = False) # 升维
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias = False) # 降维
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias = False) # 门控
        self.dropout = nn.Dropout(args.dropout) # dropout
        self.act_fn = ACT2FN[args.hidden_act]
         
    # SwiGLU
    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.up_proj(x)) * self.gate_proj(x)))