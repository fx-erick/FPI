"""
Module with different Classification Models, implemented via base classes and configured via arguments.
VGGDownBlock is probably the most important one
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.ModelUtils import get_activation, Conv2DChannelWeights, Conv2DComboLayer


# A: Residual Models (ResNet-inspired) ###########################################################################
class ResidualBlock2D(nn.Module):
    """
    2D Residual Block consisting of 2 Conv Layers with BatchNorm.
    If in_channels != out_channels or stride != 1, downsampling is
    added to the identity matrix automatically.
    :param in_channels: int
    :param channels: int
    :param stride: int
    :param activation: str
    """
    def __init__(self, in_channels, channels, stride, activation):
        super(ResidualBlock2D, self).__init__()
        if in_channels != channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(num_features=channels)
            )
        else:
            self.downsample = None

        self.a = get_activation(activation)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=channels,
                               kernel_size=(3, 3),
                               stride=(stride, stride),
                               padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

    def forward(self, x):
        out = self.a(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv2(out))
        if self.downsample is not None:
            x = self.downsample(x)
        out = self.a(out + x)
        return out


class ResCNN2Dv1(nn.Module):
    """
    CNN Model to deal with 2D inputs of variable size.
    :param num_outputs (int): if > 1 a softmax ist added to the output layer
    :param input_channels (int): number of input channels e.g. 3 for RGB image
    :param block_config (tuple of ints): defines the residual blocks in the net.
           Shape should match ((IC, OC, S), (...)) with IC = input channels, OC = output channels
           & S = stride of the first conv layer of the block
    :param activation (str): e.g. "relu", "leaky_relu"
    """
    def __init__(self,
                 input_channels,
                 num_outputs,
                 block_configs,
                 activation):

        super(ResCNN2Dv1, self).__init__()

        # store model info in model_dict
        self.model_dict = {
            "type": "ResCNN2Dv1",
            "input_channels": input_channels,
            "num_outputs": num_outputs,
            "block_configs": block_configs,
            "activation": activation
        }

        # define structure
        self.a = get_activation(activation)
        self.out_channels = block_configs[-1][1]
        self.num_outputs = num_outputs

        self.convIn = nn.Conv2d(in_channels=input_channels,
                                out_channels=block_configs[0][0],
                                kernel_size=(7, 7),
                                stride=(2, 2))
        self.bnIn = nn.BatchNorm2d(num_features=block_configs[0][0])
        self.maxpoolIn = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.rBlocks = nn.ModuleList([ResidualBlock2D(in_channels=block_configs[i][0],
                                                      channels=block_configs[i][1],
                                                      stride=block_configs[i][2],
                                                      activation=activation)
                                      for i in range(0, len(block_configs))])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=self.out_channels, out_features=num_outputs)

        # initialize conv and batchnorm layers
        if activation == "relu" or activation == "leaky_relu":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=activation)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=1.0)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=1.0)

    def get_model_dict(self):
        return self.model_dict

    def embedding(self, x):
        #print(x.shape)
        out = self.convIn(x)
        #print(out.shape)
        out = self.bnIn(out)
        #print(out.shape)
        out = self.a(out)
        #print(out.shape)
        out = self.maxpoolIn(out)
        #print(out.shape)
        for rb in self.rBlocks:
            out = rb(out)
            #print(out.shape)
        out = self.avgpool(out)
        #print(out.shape)
        out = torch.reshape(out, (-1, self.out_channels))
        #print(out.shape)
        return out

    def forward(self, x):
        out = self.embedding(x)
        out = self.fc(out)
        if self.num_outputs > 1:
            return F.softmax(out, dim=-1)
        else:
            return out
# ################################################################################################################


# B: VGG Style Models - Helpers ##################################################################################
class VGGDownBlock(nn.Module):
    """
    builds a VGG downblock (the conv layers between the maxpool layers) according to some block_config.
    the block_config should look like this (((k, k), C), ((k, k), C), ...) where k defines the kernel
    and should be either 1 or 3 and C is the nuzmber of channels.
    activation should be a str, which is handed over to get_activation()
    """
    def __init__(self, in_channels, block_config, activation, groups=None, add_weight_vectors=False, use_combo_layers=False):
        """
        Args:
            in_channels: (int)
            block_config: (tuple of tuples) see above
            activation: (str) passed to get_activation()
            groups: (tuple/list of ints) optional, if provided is used as groups in conv layers, defaults to None
            add_weight_vectors: (bool) optional, if True ModelUtils.Conv2DChannelWeights() is used instead of Conv.
            use_combo_layers: (bool) optional, if True Model.Utils.ComboLayers() are used instead onf Conv.
        """
        super(VGGDownBlock, self).__init__()

        # get activation
        self.a = get_activation(activation_name=activation)
        # extend channel info by input channels
        self.channels = [in_channels]
        for b in block_config:
            self.channels.append(b[1])
        # save kernel, stride and padding info (latter depends on kernel)
        self.kernels = [b[0] for b in block_config]
        self.strides = []
        self.paddings = []

        self.contains_weight_vectors = add_weight_vectors

        if groups is not None:
            self.groups = groups
        else:
            self.groups = [1 for i in range(0, len(block_config))]

        for k in self.kernels:
            self.strides.append((1, 1))
            if ((k[0] == 1) or (k[0] == 3)) and \
                    ((k[1] == 1) or (k[1] == 3)):
                self.paddings.append((int(k[0]/3), int(k[1]/3)))
            else:
                raise ValueError("Unexpected kernel size in VGGDonwBlock: \n" +
                                 "The method is only intended for kernels containing 3 and 1")

        if add_weight_vectors:
            self.vgg_down_block = nn.ModuleList([
                Conv2DChannelWeights(in_channels=self.channels[idx],
                                     out_channels=self.channels[idx + 1],
                                     kernel_size=self.kernels[idx],
                                     stride=self.strides[idx],
                                     padding=self.paddings[idx],
                                     groups=self.groups[idx]
                                     ) for idx in range(0, len(block_config))
            ])
        elif use_combo_layers:
            self.vgg_down_block = nn.ModuleList([
                Conv2DComboLayer(in_channels=self.channels[idx],
                                 out_channels=self.channels[idx + 1],
                                 kernel_size=self.kernels[idx],
                                 stride=self.strides[idx],
                                 padding=self.paddings[idx],
                                 groups=self.groups[idx]
                                 ) for idx in range(0, len(block_config))
            ])
        else:
            self.vgg_down_block = nn.ModuleList([
                nn.Conv2d(in_channels=self.channels[idx],
                          out_channels=self.channels[idx + 1],
                          kernel_size=self.kernels[idx],
                          stride=self.strides[idx],
                          padding=self.paddings[idx],
                          groups=self.groups[idx]
                          ) for idx in range(0, len(block_config))
            ])

    def forward(self, x):
        for vb in self.vgg_down_block:
            x = self.a(vb(x))
        return x

    def get_weight_vectors(self):
        """
        Returns: list of weight vectors in Conv2DChannelWeights()

        """
        weight_vectors = []
        for layer in self.vgg_down_block:
            if isinstance(layer, Conv2DChannelWeights):
                weight_vectors.extend(layer.get_weights())
        return weight_vectors


class VGGStandardClassifier(nn.Module):
    """
    returns a standard vgg style classifier with relu und dropout
    """
    def __init__(self, num_inputs, num_classes):
        """
        Args:
            num_inputs: (int)
            num_classes: (int)
        """
        super(VGGStandardClassifier, self).__init__()
        self.classifier = nn.ModuleList([
            nn.Linear(in_features=num_inputs, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        ])

    def forward(self, x):
        for layer in self.classifier:
            x = layer(x)
        return F.softmax(x, dim=-1)


def vgg_initialize_weights(modules):
    """
    initilaizes module acc. to pytorch implementation at:
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    """

    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


class VGGbase(nn.Module):
    """
    a VGG style classification model according to a given block config.
    config should look like this: ( ((k, k), C), (k, k), C), ...),  ... ). where (k, k) is the kernel size
    and C is the number of out_channels of the conv layer.
    After a block of conv layers follows a maxPool2D layer.
    """
    def __init__(self, input_size, block_configs, in_channels, num_classes, activation):
        """
        Args:
            input_size: (int) height or width of the square input image
            block_configs: (tuple of tuples) defines blocks, see above
            in_channels: (int)
            num_classes: (int)
            activation: (str) passed to ModelUtils.get_activation()
        """
        super(VGGbase, self).__init__()
        self.block_configs = block_configs
        self.model_dict = {
            "type": "VGGbase",
            "block_configs": block_configs,
            "input_size": input_size,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "activation": activation
        }

        # get number of hidden features for classifier
        n_last_channels = self.block_configs[-1][-1][-1]
        self.num_hidden_features = n_last_channels * int(input_size / (2 ** len(self.block_configs))) ** 2
        # define input channels
        self.in_channels = [in_channels]
        for b in block_configs:
            self.in_channels.append(b[-1][1])

        # build actual layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.down_blocks = nn.ModuleList([
            VGGDownBlock(in_channels=self.in_channels[idx],
                         block_config=block_configs[idx],
                         activation=activation)
            for idx in range(0, len(block_configs))
        ])

        # get classifier
        self.classifier = VGGStandardClassifier(num_inputs=self.num_hidden_features,
                                                num_classes=num_classes)

        # init weights acc to pytorch vgg implementation
        vgg_initialize_weights(self.modules())

    def forward(self, x):
        for db in self.down_blocks:
            x = self.pool(db(x))  # activation is applied in forward of each VGGDownBlock
            #print(x.shape)
        x = torch.flatten(x, 1)
        #print(x.shape)
        return self.classifier(x)

    def get_model_dict(self):
        return self.model_dict


# TODO: add a function which returns vgg net? how about passing to run_hyperparams() ?
def vgg_11_config():
    pass


if __name__ == "__main__":

    vgg_16 = (
        (((1, 3), 64), ((3, 1), 64), ((3, 3), 64)),
        (((3, 3), 128), ((3, 3), 128)),
        (((3, 3), 256), ((3, 3), 256), ((3, 3), 256)),
        (((3, 3), 512), ((3, 3), 512), ((3, 3), 512)),
        (((3, 3), 512), ((3, 3), 512), ((3, 3), 512))
    )

    model = VGGbase(input_size=224, block_configs=vgg_16, in_channels=3, num_classes=4, activation="relu")
    print("Model:", model)

    target = torch.empty(16, dtype=torch.long).random_(4)
    x = torch.rand((16, 3, 224, 224))
    y_hat = model(x)
    print(y_hat.shape)
    print(y_hat)

