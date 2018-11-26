import torch
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.functional import cross_entropy


class CrossEntropyWithL1Loss(_WeightedLoss):
    def __init__(self, l1_reg, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='elementwise_mean'):
        super(CrossEntropyWithL1Loss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.l1_reg = l1_reg

    def forward(self, input, target, model):
        return cross_entropy(input, target, weight=self.weight,ignore_index=self.ignore_index,
                             reduction=self.reduction) + self.l1_reg * l1_regularization(model)


def l1_regularization(model):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(torch.abs(param))
    return regularization_loss
