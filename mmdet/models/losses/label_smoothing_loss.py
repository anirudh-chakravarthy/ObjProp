# Adapted from https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module
class LabelSmoothingLoss(nn.Module):
    def __init__(self, 
                 num_classes,
                 smoothing=0.0,
                 dim=-1,
                 reduction='mean',
                 loss_weight=1.0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, 
                pred,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum') 
        reduction = (
            reduction_override if reduction_override else self.reduction)
        pred = F.log_softmax(pred, dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, label.data.unsqueeze(1), self.confidence)
        loss = torch.sum(-true_dist * pred, dim=self.dim)

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss_cls = self.loss_weight * weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return loss_cls
