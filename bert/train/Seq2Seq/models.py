import torch
from torch import nn


class Seq2Seq(nn.Module):

    def __init__(self, model, device):
        super(Seq2Seq, self).__init__()

        self.model = model
        self.device = device
        self.Linear = nn.Linear(128, 2)
        self.activation = nn.Softmax(dim=2)
        self.loss_function = nn.MSELoss()

    def forward(self, inputs, targets):

        outputs = self.model(inputs)

        # the front two ouputs is going to be ignored
        # encoded_sources: (batch_size, seq_len, embed_size)
        mlm_outputs, nsp_outputs, encoded_sources = outputs
        x = encoded_sources
        assert x.shape[2] == 128
        assert x.shape[1] == 600 == targets.shape[1] # During the test, the length is fixed to 600
        x = self.Linear(x)
        x = self.activation(x)

        loss = self.loss_function(x, targets)
        predictions = x

        return predictions, loss.unsqueeze(dim=0)
