import torch
import torch.nn as nn
import einops


class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1)  # BxNxD
        key = torch.nn.functional.normalize(key, dim=-1)  # BxNxD

        query_weight = query @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor  # BxNx1

        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1

        G = torch.sum(A * query, dim=1)  # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        )  # BxNxD

        out = self.Proj(G * key) + query  # BxNxD

        out = self.final(out)  # BxNxD

        return out


# # 输入 B N C ,  输出 B N C
# if __name__ == '__main__':
#     block = EfficientAdditiveAttnetion(64, 32).cuda()
#     input = torch.rand(3, 64 * 64, 64).cuda()
#     output = block(input)
#     print(input.size(), output.size())

import numpy as np
import torch
from torch import nn
from torch.nn import init
import math
from torch.nn import functional as F


class OutlookAttention(nn.Module):

    def __init__(self, dim, num_heads=1, kernel_size=3, padding=1, stride=1, qkv_bias=False,
                 attn_drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = self.head_dim ** (-0.5)

        self.v_pj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

        self.unflod = nn.Unfold(kernel_size, padding, stride)  # 手动卷积
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        # 映射到新的特征v
        v = self.v_pj(x).permute(0, 3, 1, 2)  # B,C,H,W
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unflod(v).reshape(B, self.num_heads, self.head_dim, self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,num_head,H*W,kxk,head_dim

        # 生成Attention Map
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # B,H,W,C
        attn = self.attn(attn).reshape(B, h * w, self.num_heads, self.kernel_size * self.kernel_size \
                                       , self.kernel_size * self.kernel_size).permute(0, 2, 1, 3,
                                                                                      4)  # B，num_head，H*W,kxk,kxk
        attn = self.scale * attn
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)

        # 获取weighted特征
        out = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size,
                                                        h * w)  # B,dimxkxk,H*W
        out = F.fold(out, output_size=(H, W), kernel_size=self.kernel_size,
                     padding=self.padding, stride=self.stride)  # B,C,H,W
        out = self.proj(out.permute(0, 2, 3, 1))  # B,H,W,C
        out = self.proj_drop(out)

        return out

# 输入 B, H, W, C,  输出 B, H, W, C
# if __name__ == '__main__':
#     block = OutlookAttention(dim=256).cuda()
#     # input = torch.rand(1, 64, 64, 512).cuda()
#     input = torch.rand(1, 128, 256, 256).cuda()
#     output = block(input)
#     print(input.size(), output.size())