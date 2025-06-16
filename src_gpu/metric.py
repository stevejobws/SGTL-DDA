# -*- coding: utf-8 -*-
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


# def accuracy_SBM(scores, targets):
    # targets = targets.cpu().numpy()
    # scores = scores.argmax(dim=-1).cpu().numpy()
    # return torch.from_numpy(confusion_matrix(targets, scores).astype('float32'))

def accuracy_SBM(scores, targets):
    scores = scores.view(-1)
    targets = targets.view(-1)
    return torch.from_numpy(confusion_matrix(targets, scores).astype('float32'))
