import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGGLoss(nn.Module):
    """VGG/Perceptual Loss

    Parameters
    ----------
    conv_index : str
        Convolutional layer in VGG model to use as perceptual output

    """
    def __init__(self, conv_index: str = '22'):
        super().__init__()
        vgg_features = torchvision.models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]

        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        self.vgg.requires_grad = False

    def forward(self, sr: torch.Tensor, hr: torch.Tensor, mask=None) -> torch.Tensor:
        """Compute VGG/Perceptual loss between Super-Resolved and High-Resolution

        Parameters
        ----------
        sr : torch.Tensor
            Super-Resolved model output tensor
        hr : torch.Tensor
            High-Resolution image tensor

        Returns
        -------
        loss : torch.Tensor
            Perceptual VGG loss between sr and hr

        """
        b, n_channels, h, w = sr.size()

        def _forward(x):
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)

        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        if mask is not None:
            mask = mask.float()
            mask = F.max_pool2d(mask, kernel_size=2)[:, 0, ...].unsqueeze(1)
            mask = mask.expand_as(vgg_sr).bool()

            loss = F.mse_loss(vgg_sr[mask], vgg_hr[mask], reduction='mean')

        else:
            loss = F.mse_loss(vgg_sr, vgg_hr, reduction='mean')

        return loss
