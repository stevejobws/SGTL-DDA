import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# 定义损失函数
bce_loss = nn.BCEWithLogitsLoss()
focal_loss = FocalLoss(alpha=1, gamma=2)

def combined_loss(inputs, targets, bce_weight=0.6, focal_weight=0.4):
    bce = bce_loss(inputs, targets)
    focal = focal_loss(inputs, targets)
    return bce_weight * bce + focal_weight * focal