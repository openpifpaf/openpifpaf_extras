"""Extras for OpenPifPaf."""

import logging
import sys

import openpifpaf

from . import _version
__version__ = _version.get_versions()['version']

from .network import basenetworks
from .network import swin_transformer
from .network import xcit

LOG = logging.getLogger(__name__)


def register():
    openpifpaf.BASE_TYPES.add(basenetworks.SwinTransformer)
    openpifpaf.BASE_TYPES.add(basenetworks.XCiT)
    openpifpaf.BASE_TYPES.add(basenetworks.EffNetV2)

    # Swin architectures: swin_t is roughly equivalent to unmodified resnet50
    openpifpaf.BASE_FACTORIES['swin_t'] = lambda: basenetworks.SwinTransformer(
        'swin_t', swin_transformer.swin_tiny_patch4_window7)
    openpifpaf.BASE_FACTORIES['swin_s'] = lambda: basenetworks.SwinTransformer(
        'swin_s', swin_transformer.swin_small_patch4_window7)
    openpifpaf.BASE_FACTORIES['swin_b'] = lambda: basenetworks.SwinTransformer(
        'swin_b', swin_transformer.swin_base_patch4_window7)
    openpifpaf.BASE_FACTORIES['swin_b_window_12'] = lambda: basenetworks.SwinTransformer(
        'swin_b_window_12', swin_transformer.swin_base_patch4_window12)
    openpifpaf.BASE_FACTORIES['swin_l'] = lambda: basenetworks.SwinTransformer(
        'swin_l', swin_transformer.swin_large_patch4_window7)
    openpifpaf.BASE_FACTORIES['swin_l_window_12'] = lambda: basenetworks.SwinTransformer(
        'swin_l_window_12', swin_transformer.swin_large_patch4_window12)
    # XCiT architectures: xcit_small_12_p16 is roughly equivalent to unmodified resnet50
    openpifpaf.BASE_FACTORIES['xcit_nano_12_p16'] = lambda: basenetworks.XCiT(
        'xcit_nano_12_p16', xcit.xcit_nano_12_p16)
    openpifpaf.BASE_FACTORIES['xcit_tiny_12_p16'] = lambda: basenetworks.XCiT(
        'xcit_tiny_12_p16', xcit.xcit_tiny_12_p16)
    openpifpaf.BASE_FACTORIES['xcit_tiny_24_p16'] = lambda: basenetworks.XCiT(
        'xcit_tiny_24_p16', xcit.xcit_tiny_24_p16)
    openpifpaf.BASE_FACTORIES['xcit_small_12_p16'] = lambda: basenetworks.XCiT(
        'xcit_small_12_p16', xcit.xcit_small_12_p16)
    openpifpaf.BASE_FACTORIES['xcit_small_24_p16'] = lambda: basenetworks.XCiT(
        'xcit_small_24_p16', xcit.xcit_small_24_p16)
    openpifpaf.BASE_FACTORIES['xcit_medium_24_p16'] = lambda: basenetworks.XCiT(
        'xcit_medium_24_p16', xcit.xcit_medium_24_p16)
    openpifpaf.BASE_FACTORIES['xcit_large_24_p16'] = lambda: basenetworks.XCiT(
        'xcit_large_24_p16', xcit.xcit_large_24_p16)
    openpifpaf.BASE_FACTORIES['xcit_nano_12_p8'] = lambda: basenetworks.XCiT(
        'xcit_nano_12_p8', xcit.xcit_nano_12_p8)
    openpifpaf.BASE_FACTORIES['xcit_tiny_12_p8'] = lambda: basenetworks.XCiT(
        'xcit_tiny_12_p8', xcit.xcit_tiny_12_p8)
    openpifpaf.BASE_FACTORIES['xcit_tiny_24_p8'] = lambda: basenetworks.XCiT(
        'xcit_tiny_24_p8', xcit.xcit_tiny_24_p8)
    openpifpaf.BASE_FACTORIES['xcit_small_12_p8'] = lambda: basenetworks.XCiT(
        'xcit_small_12_p8', xcit.xcit_small_12_p8)
    openpifpaf.BASE_FACTORIES['xcit_small_24_p8'] = lambda: basenetworks.XCiT(
        'xcit_small_24_p8', xcit.xcit_small_24_p8)
    openpifpaf.BASE_FACTORIES['xcit_medium_24_p8'] = lambda: basenetworks.XCiT(
        'xcit_medium_24_p8', xcit.xcit_medium_24_p8)
    openpifpaf.BASE_FACTORIES['xcit_large_24_p8'] = lambda: basenetworks.XCiT(
        'xcit_large_24_p8', xcit.xcit_large_24_p8)
    # Parameters for the EffNetV2 construction
    # expansion ratio, channels, number of layers of this type, stride, use squeeze+excitation
    # t, c, n, s, SE
    openpifpaf.BASE_FACTORIES['effnetv2_s'] = lambda: basenetworks.EffNetV2('effnetv2_s',
                                                [
                                                    [1, 24, 2, 1, 0],
                                                    [4, 48, 4, 2, 0],
                                                    [4, 64, 4, 2, 0],
                                                    [4, 128, 6, 2, 1],
                                                    [6, 160, 9, 1, 1],
                                                    [6, 256, 15, 2, 1],
                                                ],
                                                stride=32)
    openpifpaf.BASE_FACTORIES['effnetv2_m'] = lambda: basenetworks.EffNetV2('effnetv2_m',
                                                [
                                                    [1, 24, 3, 1, 0],
                                                    [4, 48, 5, 2, 0],
                                                    [4, 80, 5, 2, 0],
                                                    [4, 160, 7, 2, 1],
                                                    [6, 176, 14, 1, 1],
                                                    [6, 304, 18, 2, 1],
                                                    [6, 512, 5, 1, 1],
                                                ],
                                                stride=32)
    openpifpaf.BASE_FACTORIES['effnetv2_l'] = lambda: basenetworks.EffNetV2('effnetv2_l',
                                                [
                                                    [1, 32, 4, 1, 0],
                                                    [4, 64, 7, 2, 0],
                                                    [4, 96, 7, 2, 0],
                                                    [4, 192, 10, 2, 1],
                                                    [6, 224, 19, 1, 1],
                                                    [6, 384, 25, 2, 1],
                                                    [6, 640, 7, 1, 1],
                                                ],
                                                stride=32)
    openpifpaf.BASE_FACTORIES['effnetv2_xl'] = lambda: basenetworks.EffNetV2('effnetv2_xl',
                                                 [
                                                     [1, 32, 4, 1, 0],
                                                     [4, 64, 8, 2, 0],
                                                     [4, 96, 8, 2, 0],
                                                     [4, 192, 16, 2, 1],
                                                     [6, 256, 24, 1, 1],
                                                     [6, 512, 32, 2, 1],
                                                     [6, 640, 8, 1, 1],
                                                 ],
                                                 stride=32)
    openpifpaf.BASE_FACTORIES['effnetv2_s16_s'] = lambda: basenetworks.EffNetV2('effnetv2_s16_s',
                                                    [
                                                        [1, 24, 2, 1, 0],
                                                        [4, 48, 4, 2, 0],
                                                        [4, 64, 4, 2, 0],
                                                        [4, 128, 6, 2, 1],
                                                        [6, 160, 9, 1, 1],
                                                        # [6, 256, 15, -1, 1],  # -1 = dilated con
                                                    ],
                                                    stride=16)
    openpifpaf.BASE_FACTORIES['effnetv2_s16_m'] = lambda: basenetworks.EffNetV2('effnetv2_s16_m',
                                                    [
                                                        [1, 24, 3, 1, 0],
                                                        [4, 48, 5, 2, 0],
                                                        [4, 80, 5, 2, 0],
                                                        [4, 160, 7, 2, 1],
                                                        [6, 176, 14, 1, 1],
                                                        # [6, 304, 18, -1, 1], # -1 = dilated conv
                                                        # [6, 512, 5, 1, 1],
                                                    ],
                                                    stride=16)
    openpifpaf.BASE_FACTORIES['effnetv2_s16_l'] = lambda: basenetworks.EffNetV2('effnetv2_s16_l',
                                                    [
                                                        [1, 32, 4, 1, 0],
                                                        [4, 64, 7, 2, 0],
                                                        [4, 96, 7, 2, 0],
                                                        [4, 192, 10, 2, 1],
                                                        [6, 224, 19, 1, 1],
                                                        # [6, 384, 25, -1, 1],  # -1=dilated conv
                                                        # [6, 640, 7, 1, 1],
                                                    ],
                                                    stride=16)
    openpifpaf.BASE_FACTORIES['effnetv2_s16_xl'] = lambda: basenetworks.EffNetV2('effnetv2_s16_xl',
                                                     [
                                                         [1, 32, 4, 1, 0],
                                                         [4, 64, 8, 2, 0],
                                                         [4, 96, 8, 2, 0],
                                                         [4, 192, 16, 2, 1],
                                                         [6, 256, 24, 1, 1],
                                                         # [6, 512, 32, -1, 1],  # -1 = dilated c
                                                         # [6, 640, 8, 1, 1],
                                                     ],
                                                     stride=16)
    openpifpaf.BASE_FACTORIES['botnet'] = lambda: basenetworks.BotNet('botnet')

    # monkey patch for checkpoint compatibility
    openpifpaf.network.basenetworks.SwinTransformer = basenetworks.SwinTransformer
    openpifpaf.network.basenetworks.FPN = basenetworks.FPN
    sys.modules["openpifpaf.network.swin_transformer"] = swin_transformer
    LOG.info("monkey patches applied for swin_transformer")
    openpifpaf.CHECKPOINT_URLS['swin_s'] = (
        'http://github.com/dmizr/openpifpaf/releases/download/'
        'v0.12.14/swin_s_fpn_lvl_3_lr_5e-5_resumed-d286d41a.pkl')
    openpifpaf.CHECKPOINT_URLS['swin_b'] = (
        'http://github.com/dmizr/openpifpaf/releases/download/'
        'v0.12.14/swin_b_fpn_lvl_3_lr_5e-5_resumed-fa951ce0.pkl')
    openpifpaf.CHECKPOINT_URLS['swin_t_input_upsample'] = (
        'http://github.com/dmizr/openpifpaf/releases/download/'
        'v0.12.14/swin_t_input_upsample_no_fpn_lr_5e-5_resumed-e0681112.pkl')
