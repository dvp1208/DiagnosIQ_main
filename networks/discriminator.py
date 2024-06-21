import torch
import torch.nn as nn
import math

from .blocks import Normalize
from .blocks import NORM_TYPE
from .initialize import init_weights


class Discriminator(nn.Module):

    def __init__(self,
                 input_dim: int,
                 mid_dim: int,
                 output_dim: int,
                 latent_size: int,
                 ) -> None:
        super().__init__()

        n_downsample = max(int(math.log2(latent_size)) - 2, 0)

        module_list = []
        for i in range(n_downsample):
            module_list.extend([
                nn.Conv2d(input_dim, input_dim, 4, 2, 1),
                Normalize(NORM_TYPE, input_dim),
                nn.LeakyReLU(0.2),
            ])

        module_list.extend([
            nn.Flatten(),
            nn.Linear(mid_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
        ])

        self.module_list = nn.Sequential(*module_list)

        init_weights(self, 'normal')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # for f in self.module_list:
        #     x = f(x)
        # return x.squeeze(1)
        return self.module_list(x).squeeze(1)
