import torch
import torch.nn as nn
import numpy as np
from configurations import get_train_config


def get_activation(activation_name):
    """
    :param activation_name: e.g. "relu", "leaky_relu"
    :return: e.g. torch.nn.ReLu()
    """
    if activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError("unknown activation provided to get_activation()")


class Conv2DChannelWeights(nn.Module):
    """
    Aggregates a Conv2d layer with an additional weight vector with one weight per out channels.
    weights are multiplied with channels during forward pass.

    weights are initialized via numpy with uniform distribution. For deterministic weights, set numpy deterministic.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        """
        Args are passed to constructor of torch.nn.Conv2d()

        Args:
            in_channels: (int)
            out_channels: (int)
            kernel_size: (tuple or int)
            stride: (int)
            padding: (int)
            groups: (int) optional, defaults to 1
        """
        super(Conv2DChannelWeights, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups)

        np.random.seed(get_train_config()["random_seed"])  # random seed for deterministic weights
        self.weight_vector = nn.parameter.Parameter(torch.from_numpy(np.random.rand(out_channels, 1, 1)).float())

    def forward(self, x):
        x = self.conv(x)
        x = x * self.weight_vector
        return x

    def get_weights(self):
        return self.weight_vector.data


class Conv2DComboLayer(nn.Module):
    """
    A combination of multiple convolution layers, to reduce parameter size
    did not really work out ...
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(Conv2DComboLayer, self).__init__()

        if groups > 1:
            self.paths = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=int(out_channels/groups),
                              kernel_size=(1, 1),
                              stride=(1, 1),
                              padding=(0, 0),
                              groups=1),
                    nn.Conv2d(in_channels=int(out_channels/groups),
                              out_channels=int(out_channels/groups),
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=1)
                ) for i in range(0, groups)])
        else:
            self.paths = nn.ModuleList([
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          groups=1)
            ])

    def forward(self, x):
        x_paths = []
        for path_layers in self.paths:
            x_paths.append(path_layers(x))

        x = x_paths[0]
        if len(x_paths) > 1:
            for i in range(1, len(x_paths)):
                x = torch.cat((x, x_paths[i]), 1)
        return x


if __name__ == "__main__":
    module = Conv2DComboLayer(512, 512, (3, 3), (1, 1), (1, 1), 16)

    x = torch.rand(16, 512, 12, 12)
    y = module(x)
    print(y.shape)
