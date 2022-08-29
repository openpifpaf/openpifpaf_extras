import argparse
import logging
import math
import openpifpaf
import torch
import torchvision.models
from . import effnetv2
from . import bottleneck_transformer

LOG = logging.getLogger(__name__)


class FPN(torch.nn.Module):
    """ Feature Pyramid Network (https://arxiv.org/abs/1612.03144), modified to only
    refine and return the feature map of a single pyramid level.

    This implementation is more computationally efficient than torchvision's
    FeaturePyramidNetwork when only a single feature map is needed, as it avoids refining
    (i.e. applying a 3x3 conv on) feature maps that aren't used later on.

    For example, for Swin, if only the feature map of stride 8 (fpn_level=2) is needed,
    the feature maps of stride 4, 16 and 32 won't get refined with this implementation.
    """

    def __init__(self, in_channels, out_channels, fpn_level=3):

        super().__init__()

        self.num_upsample_ops = len(in_channels) - fpn_level

        self.lateral_convs = torch.nn.ModuleList()

        # Start from the higher-level features (start from the smaller feature maps)
        for i in range(1, 2 + self.num_upsample_ops):
            lateral_conv = torch.nn.Conv2d(in_channels[-i], out_channels, 1)
            self.lateral_convs.append(lateral_conv)

        # Only one single refine conv for the largest feature map
        self.refine_conv = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, inputs):
        # FPN top-down pathway
        # Start from the higher-level features (start from the smaller feature maps)
        laterals = [lateral_conv(x)
                    for lateral_conv, x in zip(self.lateral_convs, inputs[::-1])]

        for i in range(1, 1 + self.num_upsample_ops):
            laterals[i] += torch.nn.functional.interpolate(
                laterals[i - 1], size=laterals[i].shape[2:], mode='nearest')

        x = self.refine_conv(laterals[-1])
        return x


class SwinTransformer(openpifpaf.network.BaseNetwork):
    """Swin Transformer, with optional FPN and input upsampling to obtain higher resolution
    feature maps"""
    pretrained = True
    drop_path_rate = 0.2
    input_upsample = False
    use_fpn = False
    fpn_level = 3
    fpn_out_channels = None

    def __init__(self, name, swin_net):
        embed_dim = swin_net().embed_dim

        if not self.use_fpn or self.fpn_out_channels is None:
            self.out_features = 8 * embed_dim
        else:
            self.out_features = self.fpn_out_channels

        stride = 32

        if self.input_upsample:
            LOG.debug('swin input upsampling')
            stride //= 2

        if self.use_fpn:
            LOG.debug('swin output FPN level: %d', self.fpn_level)
            stride //= 2 ** (4 - self.fpn_level)

        super().__init__(name, stride=stride, out_features=self.out_features)

        self.input_upsample_op = None
        if self.input_upsample:
            self.input_upsample_op = torch.nn.Upsample(scale_factor=2)

        if not self.use_fpn:
            out_indices = [3, ]
        else:
            out_indices = list(range(self.fpn_level - 1, 4))

        self.backbone = swin_net(pretrained=self.pretrained,
                                 drop_path_rate=self.drop_path_rate,
                                 out_indices=out_indices)

        self.fpn = None
        if self.use_fpn:
            self.fpn = FPN([embed_dim, 2 * embed_dim, 4 * embed_dim, 8 * embed_dim],
                           self.out_features, self.fpn_level)

    def forward(self, x):
        if self.input_upsample_op is not None:
            x = self.input_upsample_op(x)

        outs = self.backbone(x)

        if self.fpn is not None:
            x = self.fpn(outs)
        else:
            x = outs[-1]

        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('SwinTransformer')

        group.add_argument('--swin-drop-path-rate', default=cls.drop_path_rate, type=float,
                           help="drop path (stochastic depth) rate")

        group.add_argument('--swin-input-upsample', default=False, action='store_true',
                           help='scales input image by a factor of 2 for higher res feature maps')

        group.add_argument('--swin-use-fpn', default=False, action='store_true',
                           help='adds a FPN after the Swin network '
                                'to obtain higher res feature maps')

        group.add_argument('--swin-fpn-out-channels',
                           default=cls.fpn_out_channels, type=int,
                           help='output channels of the FPN (None to use the '
                                'default number of channels of the Swin network)')

        group.add_argument('--swin-fpn-level',
                           default=cls.fpn_level, type=int,
                           help='FPN pyramid level, must be between 1 '
                                '(highest resolution) and 4 (lowest resolution)')

        group.add_argument('--swin-no-pretrain', dest='swin_pretrained',
                           default=True, action='store_false',
                           help='use randomly initialized models')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.drop_path_rate = args.swin_drop_path_rate
        cls.input_upsample = args.swin_input_upsample
        cls.use_fpn = args.swin_use_fpn
        cls.fpn_out_channels = args.swin_fpn_out_channels
        cls.fpn_level = args.swin_fpn_level
        cls.pretrained = args.swin_pretrained


class XCiT(openpifpaf.network.BaseNetwork):
    pretrained = True
    out_channels = None
    out_maxpool = False

    def __init__(self, name, xcit_net):
        embed_dim = xcit_net().embed_dim
        patch_size = xcit_net().patch_size
        has_projection = isinstance(self.out_channels, int)
        self.out_channels = self.out_channels if has_projection else embed_dim

        stride = patch_size * 2 if self.out_maxpool else patch_size

        super().__init__(name, stride=stride, out_features=self.out_channels)

        self.backbone = xcit_net(pretrained=self.pretrained)

        if has_projection:
            LOG.debug('adding output projection to %d channels', self.out_features)
            self.out_projection = torch.nn.Conv2d(
                embed_dim, self.out_features, kernel_size=1, stride=1)
        else:
            LOG.debug('no output projection')
            self.out_projection = torch.nn.Identity()

        if self.out_maxpool:
            LOG.debug('output max-pooling')
            self.out_block = torch.nn.Sequential(
                self.out_projection,
                torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            )
        else:
            self.out_block = torch.nn.Sequential(self.out_projection)

    def forward(self, x):
        x = self.backbone(x)
        x = self.out_block(x)
        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('XCiT')
        group.add_argument('--xcit-out-channels',
                           default=cls.out_channels, type=int,
                           help='number of output channels for optional projection layer '
                                '(None for no projection layer)')

        group.add_argument('--xcit-out-maxpool', default=False, action='store_true',
                           help='adds max-pooling to backbone output feature map')

        group.add_argument('--xcit-no-pretrain', dest='xcit_pretrained',
                           default=True, action='store_false',
                           help='use randomly initialized models')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.out_channels = args.xcit_out_channels
        cls.out_maxpool = args.xcit_out_maxpool
        cls.pretrained = args.xcit_pretrained


class EffNetV2(openpifpaf.network.BaseNetwork):
    def __init__(self, name, configuration, stride):
        backbone = effnetv2.EffNetV2(configuration)
        super().__init__(name, stride=stride, out_features=backbone.output_channel)
        self.backbone = backbone
        self.backbone._initialize_weights()

    def forward(self, x):
        x = self.backbone.forward(x)
        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        pass

    @classmethod
    def configure(cls, args: argparse.Namespace):
        pass


class BotNet(openpifpaf.network.BaseNetwork):
    input_image_size = 640

    def __init__(self, name, out_features=2048):
        super().__init__(name, stride=8, out_features=out_features)

        layer = bottleneck_transformer.BottleStack(
            dim=256,
            fmap_size=int(math.ceil(self.input_image_size / 4)),  # default img size is 640 x 640
            dim_out=2048,
            proj_factor=4,
            downsample=True,
            heads=4,
            dim_head=128,
            rel_pos_emb=True,
            activation=torch.nn.ReLU()
        )

        resnet = torchvision.models.resnet50()

        # model surgery
        resnet_parts = list(resnet.children())
        self.backbone = torch.nn.Sequential(
            *resnet_parts[:5],
            layer,
        )

    def forward(self, x):
        x = self.backbone.forward(x)
        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('BotNet')
        group.add_argument('--botnet-input-image-size',
                           default=cls.input_image_size, type=int,
                           help='Input image size. Needs to be the same for training and'
                           ' prediction, as BotNet only accepts fixed input sizes')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.input_image_size = args.botnet_input_image_size
