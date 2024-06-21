import torch
import torch.nn as nn

from .initialize import init_weights
from .blocks import ConvBlock
from .blocks import DownConv
from .blocks import Conv1x1


class Encoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 filters: list,
                 ) -> None:
        super().__init__()

        module_list = []
        for i, filter in enumerate(filters):
            if i == 0:
                module_list.extend([
                    nn.Conv2d(input_dim, filters[0], 3, 1, 1, bias=True),
                    DownConv(filters[0], filters[1]),
                ])

            elif i < len(filters) - 1:
                module_list.extend([
                    ConvBlock(filters[i], filters[i]),
                    DownConv(filters[i], filters[i + 1]),
                ])

            elif i == len(filters) - 1:
                module_list.extend([
                    ConvBlock(filters[i], filters[i]),
                    ConvBlock(filters[i], emb_dim),
                ])

        self.module_list = nn.Sequential(*module_list)

        init_weights(self, init_type='kaiming')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module_list(x)


class NormalEncoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 filters: list,
                 ) -> None:
        super().__init__()

        module_list = []
        for i, filter in enumerate(filters):
            if i == 0:
                module_list.extend([
                    nn.Conv2d(input_dim, filters[0], 3, 1, 1, bias=True),
                    DownConv(filters[0], filters[1]),
                ])

            elif i < len(filters) - 1:
                module_list.extend([
                    ConvBlock(filters[i], filters[i]),
                    DownConv(filters[i], filters[i + 1]),
                ])

            elif i == len(filters) - 1:
                module_list.extend([
                    ConvBlock(filters[i], filters[i]),
                    ConvBlock(filters[i], filters[i]),
                ])

        self.module_list = nn.Sequential(*module_list)

        self.final_conv1 = Conv1x1(filters[i], emb_dim)
        self.final_conv2 = Conv1x1(filters[i], emb_dim)

        init_weights(self, init_type='kaiming')

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module_list(x)

        mu = self.final_conv1(x)
        logvar = self.final_conv2(x)

        if self.training:
            z = self._reparameterize(mu, logvar)
            return z, mu, logvar

        else:
            return mu
