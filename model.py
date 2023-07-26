# from typing import List, Tuple, Dict
import torch
from torch import nn


class ResBlock(nn.Module):
    
    def __init__(self, n_in : int, n_hid : int, n_out : int) -> None:
        super().__init__()
        # 正常神经网络连接
        self.through = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.BatchNorm1d(n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid),
            nn.BatchNorm1d(n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_out),
            nn.BatchNorm1d(n_out)
        )
        # 跳过连接
        self.skip = None
        if n_in != n_out:
            self.skip = nn.Sequential(
                nn.Linear(n_in, n_out),
                nn.BatchNorm1d(n_out)
            )
        # 最后: 激活
        self.final = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, x : torch.tensor) -> torch.tensor:
        # 正常连接的输出
        out_through = self.through(x) 
        # 跳过连接的输出 
        out_skip = x
        if self.skip:
            out_skip = self.skip(x)
        # 两部分输出相加, 然后激活
        out = out_through + out_skip
        out = self.final(out)
        return out


class Model(nn.Module):

    def __init__(self, d : int) -> None:
        super().__init__()
        self.d = d
        # MLP变换维度
        self.start = nn.Sequential(
            nn.Linear(4 * d, 30),
            nn.ReLU()
        )
        # 残差块组成的网络
        self.res_blocks = nn.Sequential(
            ResBlock(30, 50, 70),
            ResBlock(70, 70, 70),
            ResBlock(70, 70, 70),
            ResBlock(70, 50, 30),
            ResBlock(30, 30, 30)
        )
        # MLP变换维度
        self.end = nn.Sequential(
            nn.Linear(30, 4 * d - 2),
            nn.Sigmoid()
        )
    
    def forward(self, x : torch.tensor) -> torch.tensor:
        bs = x.size(0)
        # (bs, 2*d, 2) -> (bs, 4*d) 
        x = x.flatten(1) 
        out = self.start(x)
        out = self.res_blocks(out)
        out = self.end(out)
        # (bs, 4*d-2) -> (bs, 2*d-1, 2)
        out = out.view(bs, 2*self.d-1, 2)
        # 强制使 lamda_i^1 + lamda_i^2 = 1
        out = out.softmax(dim=-1)
        return out
