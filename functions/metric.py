import torch
import torch.nn.functional as F
import numpy as np

try:
    from torch.distributed import all_reduce
except:
    print('torch.distributed cannot be imported.')

from .loss import OneHotEncoder
from utils import get_world_size
from utils import is_distributed


class DiceCoefficient(object):
    epsilon = 1e-5

    def __init__(self, n_classes, index_to_class_name, ignore_index=None):
        super().__init__()
        self.one_hot_encoder = OneHotEncoder(n_classes).forward
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.index_to_class_name = index_to_class_name

    def __call__(self, pred, label):
        batch_size = pred.shape[0]
        output = pred.argmax(1)

        output = self.one_hot_encoder(output)
        output = output.contiguous().view(batch_size, self.n_classes, -1)

        target = self.one_hot_encoder(label)
        target = target.contiguous().view(batch_size, self.n_classes, -1)

        assert output.shape == target.shape

        dice = {}
        for i in range(self.n_classes):
            if i == self.ignore_index:
                continue

            os = output[:, i, ...]
            ts = target[:, i, ...]

            inter = torch.sum(os * ts, dim=1)
            union = torch.sum(os, dim=1) + torch.sum(ts, dim=1)
            score = torch.sum(2.0 * inter / union.clamp(min=self.epsilon))
            score /= batch_size

            if is_distributed():
                all_reduce(score)
                ws = get_world_size()
                score /= ws

            if self.index_to_class_name:
                dice[self.index_to_class_name[i]] = score
            else:
                dice[str(i)] = score

        return dice
