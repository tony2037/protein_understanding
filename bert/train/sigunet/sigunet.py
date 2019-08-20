import torch
from torch import nn
from .models import conv1d, avg_pool, deconv1d

import math

class sigunet(nn.Module):

    def __init__(self, model, m, n, kernel_size, pool_size, threshold, device, embedded_size=128, sequence_length=96):
        super(sigunet, self).__init__()

        self.model = model
        self.m = m
        self.n = n
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.threshold = threshold
        self.device = device
        self.loss_function = nn.CrossEntropyLoss()
        self.embedded_size = embedded_size
        self.sequence_length = sequence_length
        pass1_len = sequence_length
        pass2_len = self.pool_len(pass1_len, 2, 2)
        pass3_len = self.pool_len(pass2_len, 2, 2)
        self.level_1 = [conv1d(embedded_size, m, kernel_size), conv1d(m, m, kernel_size), avg_pool(2)]
        self.level_2 = [conv1d(m, (m + n), kernel_size), conv1d((m + n), (m + n), kernel_size), avg_pool(2)]
        self.level_3 = [conv1d((m + n), (m + 2 * n), kernel_size), conv1d((m + 2 * n), (m + 2 *  n), kernel_size), avg_pool(2)]
        self.delevel_1 = [conv1d((m + 2 * n), (m + 3 * n), kernel_size), conv1d((m + 3 * n), (m + 3 * n), kernel_size),\
                          deconv1d((m + 3 * n), (m + 2 * n), pass3_len, kernel_size, 2)]
        self.delevel_2 = [conv1d((2 * m + 4 * n), (m + 2 * n), kernel_size), conv1d((m + 2 * n), (m + 2 * n), kernel_size),\
                          deconv1d((m + 2 * n), (m + n), pass2_len, kernel_size, 2)]
        self.delevel_3 = [conv1d((2 * m + 2 * n), (m + n), kernel_size), conv1d((m + n), (m + n), kernel_size),\
                          deconv1d((m + n), m, pass1_len, kernel_size, 2)]
        self.finals = [conv1d((2 * m), m, kernel_size), conv1d(m, 3, kernel_size, nn.Softmax())]
        self.Linear = nn.Linear(pass1_len, 1)

    def forward(self, inputs, targets):

        outputs = self.model(inputs)

        # the front two ouputs is going to be ignored
        # encoded_sources: (batch_size, seq_len, embed_size)
        mlm_outputs, nsp_outputs, encoded_sources = outputs
        # Permute the axis to adapt to nn.Conv1d
        # encoded_sources: (batch_size, embed_size, seq_len)
        # https://discuss.pytorch.org/t/swap-axes-in-pytorch/970/2
        sigunet_input = encoded_sources.transpose(2, 1)
        assert([_ for _ in sigunet_input.shape[1:]] == [128, 96])

        out = self.level_1[0](sigunet_input)
        pass1 = self.level_1[1](out)
        out = self.level_1[2](pass1)

        out = self.level_2[0](out)
        pass2 = self.level_2[1](out)
        out = self.level_2[2](pass2)
        
        out = self.level_3[0](out)
        pass3 = self.level_3[1](out)
        out = self.level_3[2](pass3)

        out = self.delevel_1[0](out)
        out = self.delevel_1[1](out)
        out = self.delevel_1[2](out)

        out = torch.cat([out, pass3], dim=1)

        out = self.delevel_2[0](out)
        out = self.delevel_2[1](out)
        out = self.delevel_2[2](out)

        out = torch.cat([out, pass2], dim=1)

        out = self.delevel_3[0](out)
        out = self.delevel_3[1](out)
        out = self.delevel_3[2](out)

        out = torch.cat([out, pass1], dim=1)

        out = self.finals[0](out)
        out = self.finals[1](out)

        out = self.Linear(out)
        out = out.squeeze()
        _out = out.view(targets.shape[0], -1)

        # Make it (batch_size, length, channels)
        #trans_out = out.transpose(2, 1)
        # errorenous
        #out, _ = torch.max(out, 2)
        #prediction = self.pass_threshold(trans_out)

        loss = self.loss_function(_out, targets)
        predictions = self.detect_SignalPeptides(_out)

        return predictions, loss.unsqueeze(dim=0)

    def detect_SignalPeptides(self, out):
        # 3 to 2
        predictions = out.argmax(dim=1)
        tmp = [(0., 1.)[i == 2] for i in predictions]
        predict = torch.FloatTensor(tmp)

        return predict

    def pass_threshold(self, input):
        # [batch_size, length, 1]
        predict = []
        for seq in input:
            predict.append(0.)
            consecutive = 0
            for val in seq:
                if val[2] >= self.threshold:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= 4:
                    predict[-1] = 1.
                    break
        predict = torch.FloatTensor(predict)

        return predict

    def pool_len(self, len, pool_size, stride):
        """
        Calculate the length after pooling
        """
        return math.floor((len - pool_size) / stride + 1)
