import re
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import time
import datetime
import scipy.io


import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.autograd import Variable

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True



class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        size = 20
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, size, (1, 25), (1, 1)),
            nn.Conv2d(size, size, (306, 16), (1, 1)),
            nn.BatchNorm2d(size),
            nn.ELU(),
            nn.AvgPool2d((1, 15), (1, 5)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(size, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

#实现了x+fn（x）
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

#前馈神经网络
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

#gelu激活函数
class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))

#transformer编码器模块
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,#表示输入数据的嵌入维度
                 num_heads=4,#多头自注意力的头数
                 drop_p=0.5,#多头自注意力后的 dropout 概率
                 forward_expansion=4,#前馈神经网络中间层的扩展倍数
                 forward_drop_p=0.5):#前馈神经网络中的 dropout 概率
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

#把上面的transformer编码器模块组合起来
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


#把输出转换为分类预测
class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes, device):  # 添加一个设备参数
        super().__init__()
        k = 256
        self.device = device  # 将设备保存为一个属性

        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        ).to(device)  # 将这个子模块移到设备上

        self.fc = nn.Sequential(
            nn.Linear(1360, k),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(k, 16),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(16, n_classes)
        ).to(device)  # 将这个子模块移到设备上

    def forward(self, x):
        # x = self.clshead(x)
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


#模型整体框架
class Conformer(nn.Sequential):
    def __init__(self, emb_size=20, depth=5, n_classes=2, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes, device=torch.device('cpu'))
        )


