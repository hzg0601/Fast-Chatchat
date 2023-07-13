# Code adapted from https://huggingface.co/kaiokendev/superhot-13b-8k-no-rlhf-test/blob/main/llama_rope_scaled_monkey_patch.py
"""
计算ROPE位置编码矩阵的cos_pos,sin_pos，并缓存到模型中作为常数
相比原文实现，应当是用ratio参数扩大了max_seq_len_cached，从而避免seq_len的不会超过位置编码的最大长度
"""
from functools import partial

import torch
import transformers
import transformers.models.llama.modeling_llama

# ROPE 见https://blog.csdn.net/weixin_43646592/article/details/130924280
# RoPE通过绝对位置编码的方式实现相对位置编码，综合了绝对位置编码和相对位置编码的优点。
# 主要就是对attention中的q, k向量注入了绝对位置信息，然后用更新的q,k向量做attention中的内积就会引入相对位置信息了。
# ROPE的实现原理：
# 1.根据预定义的output_dim计算衰减系数\theta_i=10000^{-2i/output_dim}，其中i \in [0,output_dim//2]
# 2.计算temp=\theta * position, 其中position=torch.arange(0,max_len)
# 3. 使用torch.sin(temp),torch.cos(temp)计算三角矩阵，各重复两次,记为rep_cos,rep_sin
# 4. q_1保持原样，q_2以-奇数，正偶数的方式排列；
# 5. q_1*rep_cos+q_2*rep_sin得到位置编码后的q,k向量,注意力机制中不对v进行操作,因此实际上是用乘法执行位置编码。

class CondenseRotaryEmbedding(torch.nn.Module):
    # 1. 按原代码ratio为一个整数，默认为4；
    # 2. 这里仅仅是将cos,sin预先存储起来，基于ROPE的注意力机制实际上还有向量q_1保持原样，q_2以-奇数，正偶数的方式排列的操作
    # 3. 
    def __init__(
        self, dim:int=768, ratio:int=4, max_position_embeddings=2048, base=10000, device=None
    ):
        super().__init__()
        # inv_freq 相当于原文中的\theta
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        # tensor注册到模型的 buffers() 属性中，并命名,该命名对应的是一个持久态，不会有梯度传播给它，
        # 但是能被模型的state_dict记录下来。可以理解为模型的常数。
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.ratio = ratio
        max_position_embeddings *= ratio
        self.max_seq_len_cached = max_position_embeddings
        # print(f"Monkey Patching condense ratio {ratio}")
        # t的每个整数都分成了ratio份
        t = (
            torch.arange(
                self.max_seq_len_cached,
                device=self.inv_freq.device,
                dtype=self.inv_freq.dtype,
            )
            / ratio
        )
        # Sums the product of the elements of the input operands along dimensions
        # specified using a notation based on the Einstein summation convention.
        # 使用Einstein求和约定进行指定张量的求和->沿除重复index以外的index进行求和
        # a_i b_j = a_{ij} ->[ a_i * b_j ]_{m,n}
        #? 相当于原文中的m\theta返回一个max_seq_len_cached * dim//2维的tensor
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        # Tensor[None,None,:,:]相当于Tensor.unsqueeze
        # cos_cached/sin_cached的维度为[1,1,max_seq_len_cached, dim],最内层每一行代表了一个位置编码向量
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        # 如果seq_len大于类的max_position_embeddings * ratio，才会触发这部分重算
        # 否则直接返回buffer的sin_cached, cos_cached,沿seq_len提取
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = (
                torch.arange(
                    self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype
                )
                / self.ratio
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer(
                "cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False
            )
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def replace_llama_with_condense(ratio):
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = partial(
        CondenseRotaryEmbedding, ratio=ratio
    )
