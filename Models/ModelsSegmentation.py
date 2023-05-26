"""
Contains Segmentation Models as well as basic block configs to configure them
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Models.ModelUtils import get_activation, Conv2DChannelWeights, Conv2DComboLayer
from Models.ModelsClassification import VGGDownBlock, vgg_initialize_weights
from configurations import get_train_config
import torchvision.models
from Models.ModelEESPNet import EESPNet, EESP
import os
from Models.cnn_utils import *


class UNetClassic(nn.Module):
    """
    This implementation deviates a bit from the classical Unet,
    regarding output and loss function and paddings in conv layers.
    Here a sigmoid and one out channel is used, orig. UNet uses multiple channels
    (one for each class) and pixel-wise softmax to obtain class probabilities
    """

    def __init__(self, in_channels):

        super(UNetClassic, self).__init__()
        # SETUP HELPERS
        self.model_dict = {
            "type": "UNetClassic",
            "in_channels": in_channels
        }

        activation = "relu"

        self.down_block_configs = (
            (((3, 3),  64), ((3, 3),  64)),
            (((3, 3), 128), ((3, 3), 128)),
            (((3, 3), 256), ((3, 3), 256)),
            (((3, 3), 512), ((3, 3), 512))
        )
        self.up_block_configs = (
            {"conv": (((3, 3), 512 + 512, 512), ((3, 3), 512, 512)), "convT": (((2, 2), 512, 256),)},
            {"conv": (((3, 3), 256 + 256, 256), ((3, 3), 256, 256)), "convT": (((2, 2), 256, 128),)},
            {"conv": (((3, 3), 128 + 128, 128), ((3, 3), 128, 128)), "convT": (((2, 2), 128, 64),)},
            {"conv": (((3, 3), 64 + 64, 64), ((3, 3), 64, 64)), "convT": None}
        )

        self.in_channels = [in_channels]  # define input channels
        for b in self.down_block_configs:
            self.in_channels.append(b[-1][1])

        if self.up_block_configs[-1]["convT"] is None:  # get number of filters of last up layer
            num_last_filters = self.up_block_configs[-1]["conv"][-1][-1]
        else:
            num_last_filters = self.up_block_configs[-1]["convT"][-1][-1]

        n_latent_in_channels = self.down_block_configs[-1][-1][-1]
        n_latent_out_channels = self.up_block_configs[0]["conv"][0][1] - n_latent_in_channels

        # BUILD LAYERS
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.down_blocks = nn.ModuleList([
            VGGDownBlock(in_channels=self.in_channels[idx],
                         block_config=self.down_block_configs[idx],
                         activation=activation,
                         add_weight_vectors=False)
            for idx in range(0, len(self.down_block_configs))
        ])

        self.latent = VGGUpBlock(block_config_dict={"conv": (((3, 3), n_latent_in_channels, n_latent_in_channels*2),
                                                             ((3, 3), n_latent_in_channels*2, n_latent_in_channels*2)),
                                                    "convT": (((2, 2), n_latent_in_channels*2, n_latent_out_channels),)},
                                 activation=activation,
                                 add_weight_vectors=False)

        self.up_blocks = nn.ModuleList([
            VGGUpBlock(block_config_dict=self.up_block_configs[i],
                       activation=activation,
                       add_weight_vectors=False)
            for i in range(0, len(self.up_block_configs))
        ])

        self.final = nn.Conv2d(in_channels=num_last_filters, out_channels=1, kernel_size=(1, 1), stride=(1, 1))

        # init weights acc to pytorch vgg implementation
        vgg_initialize_weights(self.modules())

    def forward(self, x):
        skip_connections = []
        for down_block in self.down_blocks:
            x_s = down_block(x)
            skip_connections.insert(0, x_s)
            #print("down skip", x_s.shape)
            x = self.pool(x_s)  # we have to pool since we are only using the single downblocks
            #print("down pool", x.shape)

        x = self.latent(x)
        #print("latent", x.shape)

        for i in range(0, len(self.up_blocks)):
            x = self.up_blocks[i](torch.cat([x, skip_connections[i]], dim=1))
            #print("up", x.shape)

        x = self.final(x)
        #print("out", x.shape)
        return torch.sigmoid(x)

    def get_model_dict(self):
        return self.model_dict


class miniVGGUNet(nn.Module):
    """
    a small vgg style UNet for testing purposes ...
    """
    def __init__(self, in_channels, out_channels, activation):
        """
        Args:
            in_channels: (int)
            out_channels: (int)
            activation: (str)
        """
        super(miniVGGUNet, self).__init__()
        self.model_dict = {
            "type": "miniVGGUNet",
            "in_channels": in_channels,
            "out_channels": out_channels,
            "activation": activation
        }

        self.a = get_activation(activation)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2),
                                 stride=(2, 2))

        self.conv_in_1 = nn.Conv2d(in_channels=in_channels,
                                   out_channels=16,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=1)
        # pool
        self.conv_in_21 = nn.Conv2d(in_channels=16,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1)
        self.conv_in_22 = nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1)
        # pool
        self.latent_in = nn.Conv2d(in_channels=32,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=1)
        self.laten_out = nn.ConvTranspose2d(in_channels=32,
                                            out_channels=16,
                                            kernel_size=(3, 3),
                                            stride=(2, 2),
                                            padding=1,
                                            output_padding=1)

        self.conv_out_22 = nn.Conv2d(in_channels=16+32,
                                     out_channels=32,
                                     kernel_size=(3, 3),
                                     stride=(1, 1),
                                     padding=1)
        self.conv_out_21 = nn.ConvTranspose2d(in_channels=32,
                                              out_channels=16,
                                              kernel_size=(3, 3),
                                              stride=(2, 2),
                                              padding=1,
                                              output_padding=1)

        self.conv_out_1 = nn.Conv2d(in_channels=16 + 16,
                                    out_channels=16,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1)

        self.final = nn.Conv2d(in_channels=16,
                               out_channels=out_channels,
                               kernel_size=(1, 1),
                               stride=(1, 1))

    def forward(self, x):
        x1 = self.a(self.conv_in_1(x))
        x2 = self.pool(x1)
        x2 = self.a(self.conv_in_21(x2))
        x2 = self.a(self.conv_in_22(x2))
        x_out = self.pool(x2)
        x_out = self.a(self.latent_in(x_out))
        x_out = self.a(self.laten_out(x_out))
        x_out = self.a(self.conv_out_22(torch.cat([x_out, x2], dim=1)))
        x_out = self.a(self.conv_out_21(x_out))
        x_out = self.a(self.conv_out_1(torch.cat([x_out, x1], dim=1)))
        x_out = self.final(x_out)
        return torch.sigmoid(x_out)

    def get_model_dict(self):
        return self.model_dict


class dummyModel(nn.Module):
    """ dummy model, only one conv layer ... used for debugging """
    def __init__(self):
        super(dummyModel, self).__init__()
        self.dummylayer = nn.Conv2d(in_channels=3, out_channels=1,
                                    kernel_size=(1, 1))

    def forward(self, x):
        return torch.sigmoid(self.dummylayer(x))

    def get_model_dict(self):
        return {"type": "dummyModel"}


class VGGUpBlock(nn.Module):
    """
    corresponds to the VGGDownBlock in ModelsClassification, but constructs a upblock.
    is constructed acc. to block_config_dict, that should look like e.g.
    {"conv": (((K, K), IC, C),), "convT": (((K, K), C, OC),)}, this returns a block with one conv. layer
    followed by a convTranspose layer. K is kernel size (int), IC is in_channels, C is channels and OC is out_channels.
    see also templates at bottom of file.
    """
    def __init__(self, block_config_dict, activation, groups=None, add_weight_vectors=False, use_combo_layers=False):
        """
        Args:
            block_config_dict: (dict) see above
            activation: (str) e.g. 'relu' passed to ModelUtils.get_activation()
            groups: (tuple/list of int or None) defines number of groups for each conv. layer or None if no
                    groups should be used.
            add_weight_vectors: (bool) if True ModelUtils.Conv2DChannelWeights() is used
            use_combo_layers: (bool) if True ModelUtils.ComboLayers() are used (these perform badly by the way...)
        """
        super(VGGUpBlock, self).__init__()
        self.a = get_activation(activation_name=activation)
        self.vgg_up_block = nn.ModuleList([])

        if groups is not None:
            self.groups = groups
        else:
            self.groups = [1 for i in range(0, len(block_config_dict))]

        self.contains_weight_vectors = add_weight_vectors

        if block_config_dict["conv"] is not None:
            for i in range(0, len(block_config_dict["conv"])):
                # get required padding size or throw error
                k = block_config_dict["conv"][i][0]
                if ((k[0] == 1) or (k[0] == 3)) and \
                        ((k[1] == 1) or (k[1] == 3)):
                    padding = (int(k[0] / 3), int(k[1] / 3))
                else:
                    raise ValueError("Unexpected kernel size in VGGUpBlock: " +
                                     "The method is only intended for kernels containing 3 and 1")
                # create conv layer
                if add_weight_vectors:
                    self.vgg_up_block.append(Conv2DChannelWeights(in_channels=block_config_dict["conv"][i][1],
                                                                  out_channels=block_config_dict["conv"][i][2],
                                                                  kernel_size=block_config_dict["conv"][i][0],
                                                                  stride=(1, 1), padding=padding, groups=self.groups[i]))
                elif use_combo_layers:
                    self.vgg_up_block.append(Conv2DComboLayer(in_channels=block_config_dict["conv"][i][1],
                                                              out_channels=block_config_dict["conv"][i][2],
                                                              kernel_size=block_config_dict["conv"][i][0],
                                                              stride=(1, 1), padding=padding,
                                                              groups=self.groups[i]))
                else:
                    self.vgg_up_block.append(nn.Conv2d(in_channels=block_config_dict["conv"][i][1],
                                                       out_channels=block_config_dict["conv"][i][2],
                                                       kernel_size=block_config_dict["conv"][i][0],
                                                       stride=(1, 1), padding=padding, groups=self.groups[i]))

        if block_config_dict["convT"] is not None:
            for i in range(0, len(block_config_dict["convT"])):
                # get required padding size or throw error
                k = block_config_dict["convT"][i][0]
                if ((k[0] == 1) or (k[0] == 3)) and \
                        ((k[1] == 1) or (k[1] == 3)):
                    padding = (int(k[0] / 3), int(k[1] / 3))
                elif k[0] == 2 and k[1] == 2:
                    padding = (0, 0)
                else:
                    raise ValueError("Unexpected kernel size in VGGUpBlock: " +
                                     "The method is only intended for kernels containing 3 and 1")
                # create conv layer
                self.vgg_up_block.append(nn.ConvTranspose2d(in_channels=block_config_dict["convT"][i][1],
                                                            out_channels=block_config_dict["convT"][i][2],
                                                            kernel_size=block_config_dict["convT"][i][0],
                                                            stride=(2, 2), padding=padding, output_padding=padding))

    def forward(self, x):
        for layer in self.vgg_up_block:
            x = self.a(layer(x))
        return x

    def get_weight_vectors(self):
        """
        Returns: list of weight vectors from Conv2DChannelWeights 'layers'
        """
        weight_vectors = []
        for layer in self.vgg_up_block:
            if isinstance(layer, Conv2DChannelWeights):
                weight_vectors.extend(layer.get_weights())
        return weight_vectors


class UNetVGGbase(nn.Module):
    """
    Base class for all VGG and UNet based models. The concrete architecture follows specs. provided in
    down_block_configs and up_block_configs. Please use the examples at the end of file for reference
    """
    def __init__(self, down_block_configs, up_block_configs, in_channels, activation, add_conv_channel_weights=False,
                 load_weights_from_vgg11=False):
        """
        Args:
            down_block_configs: (tuple of tuples) specifies vgg down blocks, see examples at the bottom of the file
            up_block_configs: (tuple of dicts) specifies vgg up blocks, see examples at the bottom of the file
            in_channels: (int) e.g. 3 for RGB images
            activation: (str) passed to ModelUtils.get_activation()
            add_conv_channel_weights: (bool) if True uses ModelUtils.Conv2DChannelWeights instead of conv. layers
            load_weights_from_vgg11: (bool) if True uses weights from ImageNet pretrained VGG11 in down blocks (must
                                     match config of course)
        """
        super(UNetVGGbase, self).__init__()
        # SETUP HELPERS
        self.model_dict = {
            "type": "UNetVGGbase",
            "down_block_configs": down_block_configs,
            "up_block_configs": up_block_configs,
            "in_channels": in_channels,
            "activation": activation,
            "add_conv_channel_weights": add_conv_channel_weights,
            "load_weights_from_vgg11": load_weights_from_vgg11
        }

        self.down_block_configs = down_block_configs
        self.up_block_configs = up_block_configs

        self.in_channels = [in_channels]  # define input channels
        for b in down_block_configs:
            self.in_channels.append(b[-1][1])

        if up_block_configs[-1]["convT"] is None:  # get number of filters of last up layer
            num_last_filters = up_block_configs[-1]["conv"][-1][-1]
        else:
            num_last_filters = up_block_configs[-1]["convT"][-1][-1]

        n_latent_in_channels = down_block_configs[-1][-1][-1]
        n_latent_out_channels = up_block_configs[0]["conv"][0][1] - n_latent_in_channels

        # BUILD LAYERS
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.down_blocks = nn.ModuleList([
            VGGDownBlock(in_channels=self.in_channels[idx],
                         block_config=down_block_configs[idx],
                         activation=activation,
                         add_weight_vectors=add_conv_channel_weights)
            for idx in range(0, len(down_block_configs))
        ])

        self.latent = VGGUpBlock(block_config_dict={"conv": (((3, 3), n_latent_in_channels, n_latent_in_channels),),
                                                    "convT": (((3, 3), n_latent_in_channels, n_latent_out_channels),)},
                                 activation=activation,
                                 add_weight_vectors=add_conv_channel_weights)

        self.up_blocks = nn.ModuleList([
            VGGUpBlock(block_config_dict=self.up_block_configs[i],
                       activation=activation,
                       add_weight_vectors=add_conv_channel_weights)
            for i in range(0, len(self.up_block_configs))
        ])

        self.final = nn.Conv2d(in_channels=num_last_filters, out_channels=1, kernel_size=(1, 1), stride=(1, 1))

        # init weights acc to pytorch vgg implementation
        vgg_initialize_weights(self.modules())

        # copy weights from ImageNet-pretrained VGG11 if specified
        if load_weights_from_vgg11:
            vgg_source_model = torchvision.models.vgg11(pretrained=True)

            source_vgg = []
            for m in vgg_source_model.features.modules():
                # get only the conv layers from vgg11
                if isinstance(m, nn.Conv2d):
                    source_vgg.append(m)

            target_vgg = []
            for m in self.down_blocks.modules():
                # append all conv layers in down_blocks. if down_block_configs does not match
                # vgg11, this will throw an error eventually ...
                if isinstance(m, nn.Conv2d):
                    target_vgg.append(m)

            for s, t in zip(source_vgg, target_vgg):
                if isinstance(s, nn.Conv2d) and isinstance(t, nn.Conv2d):
                    assert s.weight.size() == t.weight.size()
                    assert s.bias.size() == t.bias.size()
                    t.weight.data = s.weight.data
                    t.bias.data = s.bias.data

    def forward(self, x):
        skip_connections = []
        for down_block in self.down_blocks:
            x_s = down_block(x)
            skip_connections.insert(0, x_s)
            x = self.pool(x_s)  # we have to pool since we are only using the single downblocks

        x = self.latent(x)
        for i in range(0, len(self.up_blocks)):
            x = self.up_blocks[i](torch.cat([x, skip_connections[i]], dim=1))

        x = self.final(x)
        return torch.sigmoid(x)

    def get_model_dict(self):
        return self.model_dict

    def get_weight_vectors(self):
        """
        Returns: list of weight vectors from Conv2DChannelWeights 'layers'
        """
        weight_vectors = []
        for db in self.down_blocks:
            weight_vectors.extend(db.get_weight_vectors())
        weight_vectors.extend(self.latent.get_weight_vectors())
        for ub in self.up_blocks:
            weight_vectors.extend(ub.get_weight_vectors())
        return weight_vectors

    def set_own_weights_below_threshold_to_zero(self, threshold):
        """
        Args:
            threshold: (float) threshold value

        Returns: void
        """
        # loop through weights and set smaller than thresh to zero
        print(f"\n> pruning own weights, if abs(x) < {threshold}: x = 0.0")
        ctr_all = 0
        ctr_pruned = 0
        weights = self.get_weight_vectors()
        for vector in weights:
            for i in range(0, len(vector)):
                ctr_all += 1
                if torch.abs(vector.data[i]) < threshold:
                    vector.data[i] = 0.0
                    vector.data[i].requires_grad = False
                    ctr_pruned += 1
        print(f"  set {100.0 * (ctr_pruned / ctr_all)} percent of weights to 0.0")


class UNetVGGwcc(nn.Module):
    """
    practically works like the UNetVGGbase, but adds additional weight vectors (one weight for each put channel) after
    each vgg block. These are initialized via numpy from a uniform distribution. So for deterministic weights,
    just set numpy deterministic. For details on block_configs see impl. of UNetVGGbase.
    """
    def __init__(self, down_block_configs, up_block_configs, in_channels, activation):
        super(UNetVGGwcc, self).__init__()
        # SETUP HELPERS
        self.model_dict = {
            "type": "UNetVGGwcc",
            "down_block_configs": down_block_configs,
            "up_block_configs": up_block_configs,
            "in_channels": in_channels,
            "activation": activation
        }

        all_channels = [in_channels]  # define input channels
        for b in down_block_configs:
            all_channels.append(b[-1][1])

        if up_block_configs[-1]["convT"] is None:  # get number of filters of last up layer
            num_last_filters = up_block_configs[-1]["conv"][-1][-1]
        else:
            num_last_filters = up_block_configs[-1]["convT"][-1][-1]

        n_latent_in_channels = down_block_configs[-1][-1][-1]
        n_latent_out_channels = up_block_configs[0]["conv"][0][1] - n_latent_in_channels

        # set seed for deterministic weight vector weights
        np.random.seed(get_train_config()["random_seed"])

        # BUILD LAYERS
        self.channel_weights = nn.ParameterList([])  # weight parameters
        for db in down_block_configs:
            self.channel_weights.append(nn.parameter.Parameter(torch.from_numpy(np.random.rand(db[-1][-1], 1, 1)).float()))
        for ub in up_block_configs:
            if ub["convT"] is not None:
                self.channel_weights.append(nn.parameter.Parameter(torch.from_numpy(np.random.rand(ub["convT"][-1][-1], 1, 1)).float()))
            else:
                self.channel_weights.append(nn.parameter.Parameter(torch.from_numpy(np.random.rand(ub["conv"][-1][-1], 1, 1)).float()))

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.down_blocks = nn.ModuleList([
            VGGDownBlock(in_channels=all_channels[idx],
                         block_config=down_block_configs[idx],
                         activation=activation)
            for idx in range(0, len(down_block_configs))
        ])

        self.latent = VGGUpBlock(block_config_dict={"conv": (((3, 3), n_latent_in_channels, n_latent_in_channels),),
                                                    "convT": (((3, 3), n_latent_in_channels, n_latent_out_channels),)},
                                 activation=activation)

        self.up_blocks = nn.ModuleList([
            VGGUpBlock(block_config_dict=up_block_configs[i],
                       activation=activation)
            for i in range(0, len(up_block_configs))
        ])

        self.final = nn.Conv2d(in_channels=num_last_filters, out_channels=1, kernel_size=(1, 1), stride=(1, 1))

        # init weights acc to pytorch vgg implementation
        vgg_initialize_weights(self.modules())

    def forward(self, x):
        skip_connections = []
        ccw_ctr = 0
        for i in range(0, len(self.down_blocks)):
            for layer in self.down_blocks[i].vgg_down_block:
                x = self.down_blocks[i].a(layer(x))
            # print(x.shape, self.channel_weights[ccw_ctr].shape)
            x = x * self.channel_weights[ccw_ctr]
            ccw_ctr += 1

            skip_connections.insert(0, x)
            x = self.pool(x)  # we have to pool since we are only using downblocks

        #print(x.shape)
        x = self.latent(x)
        for i in range(0, len(self.up_blocks)):
            x = torch.cat([x, skip_connections[i]], dim=1)
            for layer in self.up_blocks[i].vgg_up_block:
                x = self.up_blocks[i].a(layer(x))
                #print(x.shape)
            x = x * self.channel_weights[ccw_ctr]
            ccw_ctr += 1
        x = self.final(x)
        #print(x.shape)
        return torch.sigmoid(x)

    def get_model_dict(self):
        return self.model_dict

    def get_channel_weights(self):
        weights = []
        for weight_param in self.channel_weights:
            weights.append(weight_param.data.tolist())
        return weights

    def set_own_weights_below_threshold_to_zero(self, threshold=0.2):
        # loop through weights and set smaller than thresh to zero
        print(f"\n> pruning own weights, if abs(x) < {threshold}: x = 0.0")
        ctr_all = 0
        ctr_pruned = 0
        for vector in self.channel_weights:
            for i in range(0, len(vector)):
                ctr_all += 1
                if torch.abs(vector.data[i]) < threshold:
                    vector.data[i] = 0.0
                    vector.data[i].requires_grad = False
                    ctr_pruned += 1
        print(f"  set {100.0 * (ctr_pruned/ctr_all)} percent of weights to 0.0")


class UNetVGGGroupConvs(nn.Module):
    """
    also works like UNetVGGbase, but gets two additional groups arguments (tuples):

    down_groups: a group argument (int) per layer (aggregated acc. to blocks)
    up_groups: a group argument per conv layer (aggregated acc. to blocks)

    latent block does not use groups.
    """
    def __init__(self, down_block_configs, up_block_configs, down_groups, up_groups, in_channels, activation, use_combo_layers=False):
        super(UNetVGGGroupConvs, self).__init__()
        # SETUP HELPERS
        self.model_dict = {
            "type": "UNetVGGGroupConvs",
            "down_block_configs": down_block_configs,
            "up_block_configs": up_block_configs,
            "down_groups": down_groups,
            "up_groups": up_groups,
            "in_channels": in_channels,
            "activation": activation,
            "use_combo_layers": use_combo_layers
        }
        self.down_block_configs = down_block_configs
        self.up_block_configs = up_block_configs

        self.in_channels = [in_channels]  # define input channels
        for b in down_block_configs:
            self.in_channels.append(b[-1][1])

        if up_block_configs[-1]["convT"] is None:  # get number of filters of last up layer
            num_last_filters = up_block_configs[-1]["conv"][-1][-1]
        else:
            num_last_filters = up_block_configs[-1]["convT"][-1][-1]

        n_latent_in_channels = down_block_configs[-1][-1][-1]
        n_latent_out_channels = up_block_configs[0]["conv"][0][1] - n_latent_in_channels

        # BUILD LAYERS
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.down_blocks = nn.ModuleList([
            VGGDownBlock(in_channels=self.in_channels[idx],
                         block_config=down_block_configs[idx],
                         activation=activation,
                         groups=down_groups[idx],
                         add_weight_vectors=False,
                         use_combo_layers=use_combo_layers)
            for idx in range(0, len(down_block_configs))
        ])

        self.latent = VGGUpBlock(block_config_dict={"conv": (((3, 3), n_latent_in_channels, n_latent_in_channels),),
                                                    "convT": (((3, 3), n_latent_in_channels, n_latent_out_channels),)},
                                 activation=activation)

        self.up_blocks = nn.ModuleList([
            VGGUpBlock(block_config_dict=self.up_block_configs[i],
                       activation=activation,
                       groups=up_groups[i],
                       add_weight_vectors=False,
                       use_combo_layers=use_combo_layers)
            for i in range(0, len(self.up_block_configs))
        ])

        self.final = nn.Conv2d(in_channels=num_last_filters, out_channels=1, kernel_size=(1, 1), stride=(1, 1))

        # init weights acc to pytorch vgg implementation
        vgg_initialize_weights(self.modules())

    def forward(self, x):
        skip_connections = []
        for down_block in self.down_blocks:
            x_s = down_block(x)
            skip_connections.insert(0, x_s)
            x = self.pool(x_s)  # we have to pool since we are only using the single downblocks

        x = self.latent(x)
        for i in range(0, len(self.up_blocks)):
            x = self.up_blocks[i](torch.cat([x, skip_connections[i]], dim=1))

        x = self.final(x)
        return torch.sigmoid(x)

    def get_model_dict(self):
        return self.model_dict

class EESPNet_Seg(nn.Module):
    def __init__(self, classes=20, s=1, pretrained=None, gpus=1):
        super().__init__()
        self.model_dict = {
            "type": "EESPNet",
            "classes": classes,
            "s": s
        }
        classificationNet = EESPNet(classes=1000, s=s)

        # load the pretrained weights
        if pretrained:
            if not os.path.isfile(pretrained):
                print('Weight file does not exist. Training without pre-trained weights')
            print('Model initialized with pretrained weights')
            classificationNet.load_state_dict(torch.load(pretrained))

        self.net = classificationNet

        del classificationNet
        # delete last few layers
        del self.net.classifier
        del self.net.level5
        del self.net.level5_0
        if s <=0.5:
            p = 0.1
        else:
            p=0.2

        self.proj_L4_C = CBR(self.net.level4[-1].module_act.num_parameters, self.net.level3[-1].module_act.num_parameters, 1, 1)
        pspSize = 2*self.net.level3[-1].module_act.num_parameters
        self.pspMod = nn.Sequential(EESP(pspSize, pspSize //2, stride=1, k=4, r_lim=7),
                PSPModule(pspSize // 2, pspSize //2))
        self.project_l3 = nn.Sequential(nn.Dropout2d(p=p), C(pspSize // 2, classes, 1, 1))
        self.act_l3 = BR(classes)
        self.project_l2 = CBR(self.net.level2_0.act.num_parameters + classes, classes, 1, 1)
        self.project_l1 = nn.Sequential(nn.Dropout2d(p=p), C(self.net.level1.act.num_parameters + classes, classes, 1, 1))

    def hierarchicalUpsample(self, x, factor=3):
        for i in range(factor):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


    def forward(self, input):
        out_l1, out_l2, out_l3, out_l4 = self.net(input, seg=True)
        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = F.interpolate(out_l4_proj, scale_factor=2, mode='bilinear', align_corners=True)
        merged_l3_upl4 = self.pspMod(torch.cat([out_l3, up_l4_to_l3], 1))
        proj_merge_l3_bef_act = self.project_l3(merged_l3_upl4)
        proj_merge_l3 = self.act_l3(proj_merge_l3_bef_act)
        out_up_l3 = F.interpolate(proj_merge_l3, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l2 = self.project_l2(torch.cat([out_l2, out_up_l3], 1))
        out_up_l2 = F.interpolate(merge_l2, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l1 = self.project_l1(torch.cat([out_l1, out_up_l2], 1))
        if self.training:
            return torch.sigmoid(F.interpolate(merge_l1, scale_factor=2, mode='bilinear',
                                 align_corners=True)), torch.sigmoid(self.hierarchicalUpsample(proj_merge_l3_bef_act))
        else:
            return torch.sigmoid(F.interpolate(merge_l1, scale_factor=2, mode='bilinear', align_corners=True))

    def get_model_dict(self):
        return self.model_dict


def vgg11_unet_configs():
    vgg_11_down = (
        (((3, 3),  64), ),
        (((3, 3), 128), ),
        (((3, 3), 256), ((3, 3), 256), ),
        (((3, 3), 512), ((3, 3), 512), ),
        (((3, 3), 512), ((3, 3), 512), )
    )
    vgg_11_up = (
        {"conv": (((3, 3), 512 + 256, 512),), "convT": (((3, 3), 512, 256),)},
        {"conv": (((3, 3), 512 + 256, 512),), "convT": (((3, 3), 512, 128),)},
        {"conv": (((3, 3), 256 + 128, 256),), "convT": (((3, 3), 256,  64),)},
        {"conv": (((3, 3),  64 + 128, 128),), "convT": (((3, 3), 128,  32),)},
        {"conv": (((3, 3),  32 +  64,  32),), "convT": None}
    )
    return vgg_11_down, vgg_11_up


def vgg16_unet_configs():
    vgg_16_down = (
        (((3, 3),  64), ((3, 3),  64)),
        (((3, 3), 128), ((3, 3), 128)),
        (((3, 3), 256), ((3, 3), 256), ((3, 3), 256)),
        (((3, 3), 512), ((3, 3), 512), ((3, 3), 512)),
        (((3, 3), 512), ((3, 3), 512), ((3, 3), 512))
    )
    vgg_16_up = (
        {"conv": (((3, 3), 512 + 256, 512), ), "convT": (((3, 3), 512, 256),)},
        {"conv": (((3, 3), 512 + 256, 512), ), "convT": (((3, 3), 512, 128),)},
        {"conv": (((3, 3), 256 + 128, 256), ), "convT": (((3, 3), 256,  64),)},
        {"conv": (((3, 3),  64 + 128, 128), ), "convT": (((3, 3), 128,  32),)},
        {"conv": (((3, 3),  32 +  64,  32), ), "convT": None}
    )
    return vgg_16_down, vgg_16_up


"""
Notes: Other kernel configurations
"""
down_all_1x3 = (
    (((1, 3),  64), ((3, 1),  64)),
    (((1, 3), 128), ((3, 1), 128)),
    (((1, 3), 256), ((3, 1), 256), ((1, 3), 256), ((3, 1), 256)),
    (((1, 3), 512), ((3, 1), 512), ((1, 3), 512), ((3, 1), 512)),
    (((1, 3), 512), ((3, 1), 512), ((1, 3), 512), ((3, 1), 512))
)

up_all_1x3 = (
    {"conv": (((1, 3), 512 + 256, 512), ((3, 1), 512, 512)), "convT": (((3, 3), 512, 256), )},
    {"conv": (((1, 3), 512 + 256, 512), ((3, 1), 512, 512)), "convT": (((3, 3), 512, 128), )},
    {"conv": (((1, 3), 256 + 128, 256), ((3, 1), 256, 256)), "convT": (((3, 3), 256,  64), )},
    {"conv": (((1, 3),  64 + 128, 128), ((3, 1), 128, 128)), "convT": (((3, 3), 128,  32), )},
    {"conv": (((1, 3),  32 +  64,  32), ((3, 1),  32,  32)), "convT": None}
)

down_outer_1x3 = (
    (((1, 3),  64), ((3, 1),  64)),
    (((1, 3), 128), ((3, 1), 128)),
    (((1, 3), 256), ((3, 1), 256), ((1, 3), 256), ((3, 1), 256)),
    (((3, 3), 512), ((3, 3), 512), ),
    (((3, 3), 512), ((3, 3), 512), )
)

up_outer_1x3 = (
    {"conv": (((3, 3), 512 + 256, 512), ), "convT": (((3, 3), 512, 256), )},
    {"conv": (((3, 3), 512 + 256, 512), ), "convT": (((3, 3), 512, 128), )},
    {"conv": (((1, 3), 256 + 128, 256), ((3, 1), 256, 256)), "convT": (((3, 3), 256, 64), )},
    {"conv": (((1, 3), 64 + 128, 128), ((3, 1), 128, 128)), "convT": (((3, 3), 128, 32), )},
    {"conv": (((1, 3), 32 + 64, 32), ((3, 1), 32, 32)), "convT": None}
)

down_inner_1x3 = (
    (((3, 3), 64), ),
    (((3, 3), 128), ),
    (((1, 3), 256), ((3, 1), 256), ((1, 3), 256), ((3, 1), 256)),
    (((1, 3), 512), ((3, 1), 512), ((1, 3), 512), ((3, 1), 512)),
    (((1, 3), 512), ((3, 1), 512), ((1, 3), 512), ((3, 1), 512))
)

up_inner_1x3 = (
    {"conv": (((1, 3), 512 + 256, 512), ((3, 1), 512, 512)), "convT": (((3, 3), 512, 256), )},
    {"conv": (((1, 3), 512 + 256, 512), ((3, 1), 512, 512)), "convT": (((3, 3), 512, 128), )},
    {"conv": (((1, 3), 256 + 128, 256), ((3, 1), 256, 256)), "convT": (((3, 3), 256, 64), )},
    {"conv": (((3, 3), 64 + 128, 128), ), "convT": (((3, 3), 128, 32),)},
    {"conv": (((3, 3), 32 + 64, 32), ), "convT": None}
)

down_all_1x3_slim = (
        (((1, 3), 64), ((3, 1), 64)),
        (((1, 3), 64), ((3, 1), 64)),
        (((1, 3), 128), ((3, 1), 128), ((1, 3), 128), ((3, 1), 128)),
        (((1, 3), 256), ((3, 1), 256), ((1, 3), 256), ((3, 1), 256)),
        (((1, 3), 512), ((3, 1), 512), ((1, 3), 512), ((3, 1), 512))
    )
up_all_1x3_slim = (
    {"conv": (((1, 3), 512 + 256, 512), ((3, 1), 512, 512)), "convT": (((3, 3), 512, 256), )},
    {"conv": (((1, 3), 256 + 256, 256), ((3, 1), 256, 256)), "convT": (((3, 3), 256, 128), )},
    {"conv": (((1, 3), 128 + 128, 128), ((3, 1), 128, 128)), "convT": (((3, 3), 128,  64), )},
    {"conv": (((1, 3), 64 + 64, 64), ((3, 1), 64, 64)), "convT": (((3, 3), 64,  32), )},
    {"conv": (((1, 3), 32 + 64, 32), ((3, 1), 32, 32)), "convT": None}
)

down_all_3x3_slim = (
        (((3, 3), 64), ),
        (((3, 3), 64), ),
        (((3, 3), 128), ((3, 1), 128), ),
        (((3, 3), 256), ((3, 1), 256), ),
        (((3, 3), 512), ((3, 1), 512), )
    )
up_all_3x3_slim = (
        {"conv": (((3, 3), 512 + 256, 512),), "convT": (((3, 3), 512, 256),)},
        {"conv": (((3, 3), 256 + 256, 256),), "convT": (((3, 3), 256, 128),)},
        {"conv": (((3, 3), 128 + 128, 128),), "convT": (((3, 3), 128,  64),)},
        {"conv": (((3, 3),  64 +  64,  64),), "convT": (((3, 3), 64,  32),)},
        {"conv": (((3, 3),  32 +  64,  32),), "convT": None}
    )

down_groups_1x3 = (
    (1, 1),
    (1, 1),
    (2, 2, 2, 2),
    (4, 4, 4, 4),
    (8, 8, 8, 8)
)
up_groups_1x3 = (
    (8, 8),
    (4, 4),
    (2, 2),
    (1, 1),
    (1, 1)
)

if __name__ == "__main__":

    input = torch.Tensor(1, 3, 512, 1024)
    net = EESPNet_Seg(classes=1, s=2)
    out_x_8 = net(input)
    print(out_x_8.size())
    '''
    x = torch.rand((4, 3, 256, 256))
    y = torch.rand((4, 1, 256, 256))

    down_all_3x3, up_all_3x3 = vgg11_unet_configs()

    model = UNetVGGbase(
        down_block_configs=down_all_3x3,
        up_block_configs=up_all_3x3,
        in_channels=3,
        activation="relu",
        add_conv_channel_weights=False,
        load_weights_from_vgg11=True
    )

    print(model)

    print(model.down_blocks[0].vgg_down_block[0].weight)

    y_hat = model(x)
    print(y_hat.shape)'''