
import torch
import torch.nn as nn

class MultiLayeredConv1d(torch.nn.Module):
    """Multi-layered conv1d for Transformer block.

    This is a module of multi-layered conv1d designed to replace position-wise feed-forward network
    in Transformer block, which is introduced in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    Args:
        in_chans (int): Number of input channels.
        hidden_chans (int): Number of hidden channels.
        kernel_size (int): Kernel size of conv1d.
        dropout_rate (float): Dropout rate.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(
        self, in_chans: int, hidden_chans: int, kernel_size: int, dropout_rate: float
    ):
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = torch.nn.Conv1d(
            hidden_chans, in_chans, 1, stride=1, padding=(1 - 1) // 2
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, *, in_chans).

        Returns:
            Tensor: Batch of output tensors (B, *, hidden_chans)

        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)

class MultiLayeredConv1d(torch.nn.Module):
    """Multi-layered conv1d for Transformer block.

    This is a module of multi-leyered conv1d designed to replace positionwise feed-forward network
    in Transforner block, which is introduced in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    Args:
        in_chans (int): Number of input channels.
        hidden_chans (int): Number of hidden channels.
        kernel_size (int): Kernel size of conv1d.
        dropout_rate (float): Dropout rate.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(
        self, in_chans: int, hidden_chans: int, kernel_size=5, dropout_rate=0.0,
    ):
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = torch.nn.Conv1d(
            hidden_chans, in_chans, 1, stride=1, padding=(1 - 1) // 2
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, *, in_chans).

        Returns:
            Tensor: Batch of output tensors (B, *, hidden_chans)

        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)

class Swish(torch.nn.Module):
    """
    Construct an Swish activation function for Conformer.
    """

    def forward(self, x):
        """
        Return Swish activation function.
        """
        return x * torch.sigmoid(x)
class ConvolutionModule(nn.Module):
    """
    ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernel size of conv layers.

    """

    def __init__(self, channels, kernel_size, activation=Swish(), ignore_prefix_len=0, bias=True):
        super(ConvolutionModule, self).__init__()
        # kernel_size should be an odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias, )
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=channels, bias=bias, )
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias, )
        self.activation = activation
        self.ignore_prefix_len = ignore_prefix_len

    def forward(self, x):
        """
        Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x_sub = self.depthwise_conv(x[..., self.ignore_prefix_len:])
        x_sub = self.activation(self.norm(x_sub))
        x_pre = x[..., :self.ignore_prefix_len]
        # x = self.depthwise_conv(x)
        # x = self.activation(self.norm(x))
        x = torch.cat([x_pre, x_sub], dim=-1)

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)