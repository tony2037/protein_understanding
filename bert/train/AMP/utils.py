from torch import nn
import torch

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, classification_report

class evaluator():

    def __init__(self):
        self.predictions = np.array([], dtype=np.int32)
        self.targets = np.array([], dtype=np.int32)

    def MCC(self, predictions=None, targets=None):
        if predictions is not None and targets is not None:
            self.predictions = np.concatenate((self.predictions, predictions), axis=0)
            self.targets = np.concatenate((self.targets, targets), axis=0)
        return matthews_corrcoef(self.targets, self.predictions)
        
def MCC(predictions, targets):
    print(predictions)
    print(targets)
    print(matthews_corrcoef(targets, predictions))
    return matthews_corrcoef(targets, predictions)

def ACC(predictions, targets):
    """
    TODO
    """
    return np.array(0.)

def ROC(predictions, targets):
    """
    TODO
    """
    return np.array(0.)
