from torch import nn
import torch

class concate(nn.Module):

    def __init__(self, output_size=32, hidden_size=128, onehot_size=29):
        super(concate, self).__init__()
        self.output_size = output_size
        self.onehot_size = onehot_size
        self.Weights_LM = nn.Linear(hidden_size, output_size, bias=False)
        self.Weights_x = nn.Linear(onehot_size, output_size, bias=False)
        self.Bias = nn.Parameter(torch.randn(1, output_size))

    def forward(self, inputs):
        Bert_out, onehot = inputs
        out_0 = self.Weights_LM(Bert_out)
        out_1 = self.Weights_x(onehot)

        return out_0 + out_1 + self.Bias.expand(Bert_out.shape[0], Bert_out.shape[1], self.output_size)
