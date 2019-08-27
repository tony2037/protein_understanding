import torch
from torch import nn


class Seq2Seq(nn.Module):

    def __init__(self, model, device):
        super(Seq2Seq, self).__init__()

        self.model = model
        self.device = device

    def forward(self, input):

        return
