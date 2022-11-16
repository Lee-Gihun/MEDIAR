import torch
import torch.nn as nn
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from segmentation_models_pytorch import MAnet
from segmentation_models_pytorch.base.modules import Activation

__all__ = ["MEDIARFormer"]


class MEDIARFormer(MAnet):
    """MEDIAR-Former Model"""

    def __init__(
        self,
        encoder_name="mit_b5",
        encoder_weights="imagenet",
        decoder_channels=(1024, 512, 256, 128, 64),
        decoder_pab_channels=256,
        in_channels=3,
        classes=3,
    ):
        super(MEDIARFormer, self).__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            decoder_pab_channels=decoder_pab_channels,
            in_channels=in_channels,
            classes=classes,
        )

        # Delete MAnet Head
        self.segmentation_head = None

        # Convert all Encoder/Decoder activations to 0
        convert_relu_to_mish(self.encoder)
        convert_relu_to_mish(self.decoder)

        self.cellprob_head = DeepSegmantationHead(
            in_channels=decoder_channels[-1], out_channels=1, kernel_size=3,
        )
        self.gradflow_head = DeepSegmantationHead(
            in_channels=decoder_channels[-1], out_channels=2, kernel_size=3,
        )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        gradflow_mask = self.gradflow_head(decoder_output)
        cellprob_mask = self.cellprob_head(decoder_output)

        masks = torch.cat([gradflow_mask, cellprob_mask], dim=1)

        return masks


class DeepSegmantationHead(nn.Sequential):
    """SegmentationHead for Cell Probability & Grad Flows"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        conv2d_1 = nn.Conv2d(
            in_channels,
            in_channels // 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        bn = nn.BatchNorm2d(in_channels // 2)
        conv2d_2 = nn.Conv2d(
            in_channels // 2,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        mish = nn.Mish(inplace=True)

        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        activation = Activation(activation)
        super().__init__(conv2d_1, mish, bn, conv2d_2, upsampling, activation)


def convert_relu_to_mish(model):
    """Convert ReLU atcivation to Mish"""
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Mish(inplace=True))
        else:
            convert_relu_to_mish(child)
