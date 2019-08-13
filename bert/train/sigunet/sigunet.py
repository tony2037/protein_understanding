import torch
from torch import nn
from .utils import conv1d, avg_pool, deconv1d, conv1x1_softmax


class sigunet(nn.Module):

    def __init__(self, model, m, n, kernel_size, pool_size, threshold):
        super(sigunet, self).__init__()

        self.model = model
        self.m = m
        self.n = n
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.threshold = threshold
        self.loss_functioin = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):

        outputs = self.model(inputs)

        # the front two ouputs is going to be ignored
        # encoded_sources: (batch_size, seq_len, embed_size)
        mlm_outputs, nsp_outputs, encoded_sources = outputs
        # Permute the axis to adapt to nn.Conv1d
        # encoded_sources: (batch_size, embed_size, seq_len)
        # https://discuss.pytorch.org/t/swap-axes-in-pytorch/970/2
        sigunet_input = encoded_sources.permute(0, 2, 1)

        out = conv1d(sigunet_input, channels=self.m, kernel_size=self.kernel_size)
        pass1 = conv1d(out, channels=self.m, kernel_size=self.kernel_size)
        out = avg_pool(pass1, pool_size=2)

        out = conv1d(out, channels=(self.m + self.n), kernel_size=self.kernel_size)
        pass2 = conv1d(out, channels=(self.m + self.n), kernel_size=self.kernel_size)
        out = avg_pool(pass2, pool_size=2)

        out = conv1d(out, channels=(self.m + 2 * self.n), kernel_size=self.kernel_size)
        pass3 = conv1d(out, channels=(self.m + 2 * self.n), kernel_size=self.kernel_size)
        out = avg_pool(pass3, pool_size=2)

        out = conv1d(out, channels=(self.m + 3 * self.n), kernel_size=self.kernel_size)
        out = conv1d(out, channels=(self.m + 3 * self.n), kernel_size=self.kernel_size)
        out = deconv1d(out, channels=(self.m + 2 * self.n), out_length=pass3.shape[2], kernel_size=self.kernel_size, stride=2)

        out = torch.cat((out, pass3), 1)

        out = conv1d(out, channels=(self.m + 2 * self.n), kernel_size=self.kernel_size)
        out = conv1d(out, channels=(self.m + 2 * self.n), kernel_size=self.kernel_size)
        out = deconv1d(out, channels=(self.m + self.n), out_length=pass2.shape[2], kernel_size=self.kernel_size, stride=2)

        out = torch.cat((out, pass2), 1)

        out = conv1d(out, channels=(self.m + self.n), kernel_size=self.kernel_size)
        out = conv1d(out, channels=(self.m + self.n), kernel_size=self.kernel_size)
        out = deconv1d(out, channels=(self.m), out_length=pass1.shape[2], kernel_size=self.kernel_size, stride=2)

        out = torch.cat((out, pass1), 1)

        out = conv1d(out, channels=self.m, kernel_size=self.kernel_size)
        out = conv1d(out, channels=self.m, kernel_size=self.kernel_size)
        out = conv1d(out, channels=3, kernel_size=1, act=nn.Softmax)

        # Make it (batch_size, length, channels)
        out = out.permute(0, 2, 1)
        # errorenous
        out, _ = torch.max(out, 2)

        loss = self.loss_function(out, targets)

        return out, loss.unsqueeze(dim=0)

    def pass_threshold(self, input):
        # [batch_size, length, 1]
        predict = []
        for seq in input:
            predict.append(0)
            consecutive = 0
            for val in seq:
                if val >= self.threshold:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= 4:
                    predict[-1] = 1
                    break

        return torch.tensor(predict)
