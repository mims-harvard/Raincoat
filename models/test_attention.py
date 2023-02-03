# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 21:12:40 2022

@author: Huan
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
# 加性模型
import numpy as np

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
            q: Queries张量，形状为[B, L_q, D_q]
            k: Keys张量，形状为[B, L_k, D_k]
            v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
            scale: 缩放因子，一个浮点标量
            attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
            上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        # if scale:
        #     attention = attention * scale
        # if attn_mask:
        #     # 给需要mask的地方设置一个负无穷
        #     attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention
    
    
attention_1 = ScaledDotProductAttention()
q = torch.randn((32, 128, 10))
k = torch.randn((32, 65, 10))
v = torch.randn((32, 65, 10))
out, attn = attention_1(q, k, v)
print(out.shape)
print(attn.shape)