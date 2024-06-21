import torch
import torch.nn as nn
import torchvision.models as vision_models


class ResNet(nn.Module):

    def __init__(self,
                 model_name: str,
                 fine_tuning: bool,
                 output_dim : int = 1,
                 extracted_layers=['avgpool'],
                ):
        super().__init__()

        assert model_name in {'resnet101', 'resnet152'}
        self.resnet_model = getattr(vision_models, model_name)(pretrained=True)

        if fine_tuning:
            lin = self.resnet_model.fc
            self.resnet_model.fc = nn.Linear(lin.in_features, output_dim)

        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []

        for name, module in self.resnet_model._modules.items():
            x = module(x)

            if name == 'avgpool':
                x = torch.flatten(x, 1)

            if name in self.extracted_layers:
                outputs.append(x)

        return x, outputs
