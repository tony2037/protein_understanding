from torch import nn
import torch

class conv1d(nn.Module):

    def __init__(self, in_channels, out_channels=None, kernel_size=None, act=nn.ReLU()):
        super(conv1d, self).__init__()
        """1d convoluation operation.

        Arguments:
            input (torch.Tensor): Input tensor to do convoluation.
                                                Its shape need to be `(batch_size, channels, length)`.
            in_channels (int): Number of the input channels.
            out_channels (int): Number of the output channels.
            kernel_size (int): Length of 1d kernel.
            act (activation function): Activation function after convoluation. Default is `nn.ReLU`.

        Returns:
            Tensor
        """

        assert (kernel_size & 1) == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Since there is no padding='SAME' such feature, refer to https://discuss.pytorch.org/t/convolution-1d-and-simple-function/11606/5
        # which says that by setting padding=(kernel_size // 2) can nail same objective provided kernel size is odd
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(self.kernel_size // 2))
        self.act = act

    def forward(self, input):

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

    def __init__(self, in_channels, out_channels, out_length, kernel_size, stride=1, act=nn.ReLU()):
        super(deconv1d, self).__init__()
        """1d deconvoluation operation.

        Arguments:
            input (torch.Tensor): Input tensor to do convoluation.
                                                Its shape need to be `(batch_size, channels, length)`.
            in_channels (int): Number of the input channels.
            out_channels (int): Number of the output channels.
            out_length (int): The output length
            kernel_size (int): Length of 1d kernel.
            act (activation function): Activation function after convoluation. Default is `nn.ReLU`.

        Returns:
            Tensor
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_length = out_length
        self.kernel_size = kernel_size
        self.stride = stride
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, self.stride)
        self.act = act

    def forward(self, input):
        in_length = int(input.shape[2])

        # Refer to https://pytorch.org/docs/stable/nn.html#convtranspose1d
        # Lo = (Lin - 1) * stride - 2 * padding + kernel_size + output_padding
        assert (in_length - 1) * self.stride + self.kernel_size > self.out_length
        output_padding = (in_length - 1) * self.stride + self.kernel_size - self.out_length & 1
        padding = ((in_length - 1) * self.stride + self.kernel_size - self.out_length + output_padding) / 2
        output_padding = int(output_padding)
        padding = int(padding)

        deconv_weight = self.deconv.weight
        deconv_bias = self.deconv.bias
        self.deconv = nn.ConvTranspose1d(self.in_channels, self.out_channels, self.kernel_size, self.stride, padding, output_padding)
        self.deconv.weight = deconv_weight
        self.deconv.bias = deconv_bias

        out = self.deconv(input)
        out = self.act(out)

        return out

