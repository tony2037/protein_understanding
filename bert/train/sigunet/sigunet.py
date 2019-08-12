from torch import nn
from .utils import conv1d, avg_pool, deconv1d, conv1x1_softmax


class sigunet(nn.Module):

    def __init__(self, model, m, n, ConvKernel_size, PoolingKernel_size):
        super(unet, self).__init__()

        self.model = model
        self.m = m
        self.n = n
        self.ConvKernel_size = ConvKernel_size
        self.PoolingKernel_size = PoolingKernel_size
        self.Pooling = nn.AvgPool1d(self.PoolingKernel_size, stride = 1)
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
        level_1 = nn.Conv1d(sigunet_input.shape[1], self.m, self.ConvKernel_size)(sigunet_input)
        level_1 = nn.Conv1d(self.m, self.m, self.ConvKernel_size)(level_1)

        level_2 = self.Pooling(level_1)
        level_2 = nn.Conv1d(level_2.shape[1], self.m + self.n, self.ConvKernel_size)(level_2)
        level_2 = nn.Conv1d(self.m + self.n, self.m + self.n, self.ConvKernel_size)(level_2)

        level_3 = self.Pooling(level_2)
        level_3 = nn.Conv1d(level_3.shape[1], self.m + 2 * self.n, self.ConvKernel_size)(level_3)
        level_3 = nn.Conv1d(self.m + 2 * self.n, self.m + 2 * self.n, self.ConvKernel_size)(level_3)

        level_4 = self.Pooling(level_3)
        level_4 = nn.Conv1d(level_4.shape[1], self.m + 3 * self.n, self.ConvKernel_size)(level_4)
        level_4 = nn.Conv1d(self.m + 3 * self.n, self.m + 3 * self.n, self.ConvKernel_size)(level_4)

        # Not sure about the kernel size applied here
        relevel_3 = nn.ConvTranspose1d(level_4[1], self.m + 2 * self.n, self.PoolingKerne_size)(level_4)
        relevel_3 = relevel_3 + level_3
        relevel_3 = nn.Conv1d(2 * self.m + 4 * self.n, self.m + 2 * self.n, self.ConvKernel_size)(relevel_3)
        relevel_3 = nn.Conv1d(self.m + 2 * self.n, self.m + 2 * self.n, self.ConvKernel_size)(relevel_3)

        relevel_2 = nn.ConvTranspose1d(relevel_3[1], self.m +  * self.n, self.PoolingKerne_size)(relevel_3)
        relevel_2 = relevel_2 + level_2
        relevel_2 = nn.Conv1d(2 * self.m + 2 * self.n, self.m + self.n, self.ConvKernel_size)(relevel_2)
        relevel_2 = nn.Conv1d(self.m + self.n, self.m + self.n, self.ConvKernel_size)(relevel_2)

        relevel_1 = nn.ConvTranspose1d(relevel_2[1], self.m, self.PoolingKerne_size)(relevel_2)
        relevel_1 = relevel_1 + level_1
        relevel_1 = nn.Conv1d(2 * self.m, self.m, self.ConvKernel_size)(relevel_1)
        relevel_1 = nn.Conv1d(self.m, self.m, self.ConvKernel_size)(relevel_1)
        outputs = nn.Conv1d(self.m, 3, 1)(relevel_1)

        loss = self.loss_function(outputs, targets)

        return predictions, loss.unsqueeze(dim=0)
