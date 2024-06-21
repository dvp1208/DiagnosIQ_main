import torch
import torch.nn as nn
import torch.nn.functional as F

from .initialize import init_weights
from .blocks import Conv1x1
from .blocks import ConvBlock
from .blocks import UpConv
from .blocks import Normalize


class Decoder(nn.Module):

    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 filters: list,
                 ) -> None:
        super().__init__()

        module_list = []
        for i, filter in enumerate(filters):
            if i == 0:
                module_list.extend([
                    nn.Conv2d(emb_dim, filters[0], 3, 1, 1),
                    ConvBlock(filters[0], filters[0]),
                ])

            elif i < len(filters) - 1:
                module_list.extend([
                    UpConv(filters[i], filters[i + 1]),
                    ConvBlock(filters[i + 1], filters[i + 1]),
                ])

            elif i == len(filters) - 1:
                module_list.extend([
                    UpConv(filters[i], filters[i]),
                    Conv1x1(filters[i], output_dim),
                ])

        self.module_list = nn.Sequential(*module_list)

        init_weights(self, init_type='kaiming')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module_list(x)
