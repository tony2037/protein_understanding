from torch import nn
import torch

import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score


def conv1d(input, channels=None, kernel_size=None, act=nn.ReLU()):
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
    in_channels = int(input.shape[1])
    # Since there is no padding='SAME' such feature, refer to https://discuss.pytorch.org/t/convolution-1d-and-simple-function/11606/5
    # which says that by setting padding=(kernel_size // 2) can nail same objective provided kernel size is odd
    conv1d = nn.Conv1d(in_channels, channels, kernel_size=kernel_size, stride=1, padding=(kernel_size // 2)).cuda()(input)
    activate = act(conv1d)

    return activate

def avg_pool(input, pool_size=2, strides=2):
    """1d average pooling operation.

    Arguments:
        input (torch.Tensor): Input tensor to do convoluation.
                                            Its shape need to be `(batch_size, channels, length)`.
        pool_size (int): Length of a pooling. Default is `2`.
        strides (int): Distance between two pooling steps. Default is `2`.

    Returns:
        Tensor
    """
    return nn.AvgPool1d(pool_size, stride=strides)(input)

def deconv1d(input, channels, out_length, kernel_size, stride=1, act=nn.ReLU()):
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

    in_length = int(input.shape[2])
    in_channels = int(input.shape[1])

    # Refer to https://pytorch.org/docs/stable/nn.html#convtranspose1d
    # Lo = (Lin - 1) * stride - 2 * padding + kernel_size + output_padding
    assert (in_length - 1) * stride + kernel_size > out_length
    output_padding = (in_length - 1) * stride + kernel_size - out_length & 1
    padding = ((in_length - 1) * stride + kernel_size - out_length + output_padding) / 2
    output_padding = int(output_padding)
    padding = int(padding)

    out = nn.ConvTranspose1d(in_channels, channels, kernel_size, stride, padding, output_padding).cuda()(input)
    out = act(out)

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
    targets = targets.detach().cpu().numpy()
    return matthews_corrcoef(targets, predictions)
    
def Signalpeptides_F1(predictions, targets):
    targets = targets_modify(targets)
    targets = targets.detach().cpu().numpy()
    return f1_score(targets, predictions)

def targets_modify(targets):
    """
    Since 2 represents containing signal peptides, which is positive sample
    """
    tmp = torch.zeros(targets.shape).cuda()
    for i in range(0, targets.shape[0]):
        if targets[i] == 2:
            tmp[i] = 1
    return tmp
