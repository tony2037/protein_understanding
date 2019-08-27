import torch
from torch import nn


class Seq2Seq(nn.Module):

    def __init__(self, model):
        super(Seq2Seq, self).__init__()

        self.model = model

    def forward(self, input):

        return
