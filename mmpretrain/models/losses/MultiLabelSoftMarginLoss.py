# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from mmengine.model import BaseModule
from torch import nn
import torch.nn.functional as F


from mmpretrain.registry import MODELS
from torch.nn import MultiLabelSoftMarginLoss
from typing import Callable, Optional
from torch import Tensor

@MODELS.register_module()
class MultiLabelSoftMarginLoss_1(MultiLabelSoftMarginLoss):

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean', ) -> None:
        super(MultiLabelSoftMarginLoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, avg_factor=None) -> Tensor:
        input
        return F.multilabel_soft_margin_loss(input, target, weight=self.weight, reduction=self.reduction)