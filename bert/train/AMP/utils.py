from torch import nn
import torch

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, classification_report

def MCC(predictions, targets):
    return matthews_corroef(targets, predictions)

def ACC(predictions, targets):
    """
    TODO
    """
    return np.array(0.)

def ROC(predictions, targets):
    """
    TODO
