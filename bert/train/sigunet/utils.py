from torch import nn
import torch

import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score


class conv1d(nn.Module):

    def __init__(self, channels=None, kernel_size=None, act=nn.ReLU()):
        super(conv1d, self).__init__()
        """1d convoluation operation.

        Arguments:
            input (torch.Tensor): Input tensor to do convoluation.
                                                Its shape need to be `(batch_size, channels, length)`.
            channels (int): Number of channels.
            kernel_size (int): Length of 1d kernel.
            act (activation function): Activation function after convoluation. Default is `nn.ReLU`.

        Returns:
            Tensor
        """

        assert (kernel_size & 1) == 1
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = None
        self.act = act
    def forward(self, input):

        in_channels = int(input.shape[1])
        # Since there is no padding='SAME' such feature, refer to https://discuss.pytorch.org/t/convolution-1d-and-simple-function/11606/5
        # which says that by setting padding=(kernel_size // 2) can nail same objective provided kernel size is odd
        if self.conv is not None:
            conv_weight = self.conv.weight
            conv_bias = self.conv.bias
            self.conv = nn.Conv1d(in_channels, self.channels, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size // 2)).cuda()
            self.conv.weight = conv_weight
            self.conv.bias = conv_bias
        else:
            self.conv = nn.Conv1d(in_channels, self.channels, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size // 2)).cuda()

        out = self.conv(input)
        out = self.act(out)
        return out


class avg_pool(nn.Module):

    def __init__(self, pool_size=2, strides=2):
        super(avg_pool, self).__init__()
        """1d average pooling operation.

        Arguments:
            input (torch.Tensor): Input tensor to do convoluation.
                                                Its shape need to be `(batch_size, channels, length)`.
            pool_size (int): Length of a pooling. Default is `2`.
            strides (int): Distance between two pooling steps. Default is `2`.

        Returns:
            Tensor
        """
        self.pool_size = pool_size
        self.strides = strides
        self.pool = nn.AvgPool1d(pool_size, stride=strides)

    def forward(self, input):

        out = self.pool(input)
        return out


class deconv1d(nn.Module):

    def __init__(self, channels, out_length, kernel_size, stride=1, act=nn.ReLU()):
        super(deconv1d, self).__init__()
        """1d deconvoluation operation.

        Arguments:
            input (torch.Tensor): Input tensor to do convoluation.
                                                Its shape need to be `(batch_size, channels, length)`.
            channels (int): Number of channels.
            out_length (int): The output length
            kernel_size (int): Length of 1d kernel.
            act (activation function): Activation function after convoluation. Default is `nn.ReLU`.

        Returns:
            Tensor
        """
        self.channels = channels
        self.out_length = out_length
        self.kernel_size = kernel_size
        self.stride = stride
        self.deconv = None
        self.act = act

    def forward(self, input):
        in_length = int(input.shape[2])
        in_channels = int(input.shape[1])

        # Refer to https://pytorch.org/docs/stable/nn.html#convtranspose1d
        # Lo = (Lin - 1) * stride - 2 * padding + kernel_size + output_padding
        assert (in_length - 1) * self.stride + self.kernel_size > self.out_length
        output_padding = (in_length - 1) * self.stride + self.kernel_size - self.out_length & 1
        padding = ((in_length - 1) * self.stride + self.kernel_size - self.out_length + output_padding) / 2
        output_padding = int(output_padding)
        padding = int(padding)

        if self.deconv is not None:
            deconv_weight = self.deconv.weight
            deconv_bias = self.deconv.bias
            self.deconv = nn.ConvTranspose1d(in_channels, self.channels, self.kernel_size, self.stride, padding, output_padding).cuda()
            self.deconv.weight = deconv_weight
            self.deconv.bias = deconv_bias
        else:
            self.deconv = nn.ConvTranspose1d(in_channels, self.channels, self.kernel_size, self.stride, padding, output_padding).cuda()

        out = self.deconv(input)
        out = self.act(out)

        return out


def conv1x1_softmax(input):
    """1d 1x1 convoluation operation. activation function is `nn.Softmax`

    Arguments:
        input (torch.Tensor): Input tensor to do convoluation.
                                            Its shape need to be `(batch_size, channels, length)`.
    Returns:
        Tensor
    """

    in_channels = int(input.shape[1])
    # Since there is no padding='SAME' such feature, refer to https://discuss.pytorch.org/t/convolution-1d-and-simple-function/11606/5
    # which says that by setting padding=(kernel_size // 2) can nail same objective provided kernel size is odd
    out = nn.Conv1d(in_channels, 3, kernel_size=1, stride=1).cuda()(input)
    activate = nn.Softmax()(out)

    return activate

def Signalpeptides_MCC(predictions, targets):
    targets = targets_modify(targets)
    # targets = targets.detach().cpu().numpy()
    return matthews_corrcoef(targets, predictions)

def Signalpeptides_F1(predictions, targets):
    targets = targets_modify(targets)
    # targets = targets.detach().cpu().numpy()
    return f1_score(targets, predictions)

def targets_modify(targets):
    """
    Since 2 represents containing signal peptides, which is positive sample
    """
    """
    tmp = torch.zeros(targets.shape)
    for i in range(0, targets.shape[0]):
        if targets[i] == 2:
            tmp[i] = 1
    return tmp
    """
    tmp = []
    for seq in targets:
        tmp.append(1. if 2 in seq else 0.)
    return np.array(tmp)
