"""
3D U-Net implementation for medical image segmentation.
Reference: https://arxiv.org/pdf/1606.06650.pdf
Adapted from pytorch-3dunet project.

Structure aligned with SFDA-DDFP project for comparable parameter count.
Default channel configuration: [64, 128, 256, 512, 512] with BatchNorm.
"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


# Default channel configuration 
DDFP_ALIGNED_F_MAPS = [64, 128, 256, 512, 512]


def number_of_features_per_level(init_channel_number: int, num_levels: int) -> list:
    """Computes the number of features at each level of the UNet"""
    return [init_channel_number * 2**k for k in range(num_levels)]


def create_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    order: str,
    num_groups: int,
    padding: int,
    dropout_prob: float,
    is3d: bool = True,
) -> list:
    """
    Create a list of modules for a given level of UNet network.
    
    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: size of the convolving kernel
        order: order of layers, e.g. 'gcr' -> groupnorm + conv + ReLU
        num_groups: number of groups for the GroupNorm
        padding: zero-padding added to all three sides of the input
        dropout_prob: dropout probability
        is3d: if True use Conv3d, otherwise use Conv2d
    """
    assert "c" in order, "Conv layer MUST be present"
    assert order[0] not in "rle", "Non-linearity cannot be the first operation"

    modules = []
    for i, char in enumerate(order):
        if char == "r":
            modules.append(("ReLU", nn.ReLU(inplace=True)))
        elif char == "l":
            modules.append(("LeakyReLU", nn.LeakyReLU(inplace=True)))
        elif char == "e":
            modules.append(("ELU", nn.ELU(inplace=True)))
        elif char == "c":
            bias = not ("g" in order or "b" in order)
            if is3d:
                conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            modules.append(("conv", conv))
        elif char == "g":
            is_before_conv = i < order.index("c")
            num_channels = in_channels if is_before_conv else out_channels
            if num_channels < num_groups:
                num_groups = 1
            assert num_channels % num_groups == 0
            modules.append(("groupnorm", nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == "b":
            is_before_conv = i < order.index("c")
            bn = nn.BatchNorm3d if is3d else nn.BatchNorm2d
            if is_before_conv:
                modules.append(("batchnorm", bn(in_channels)))
            else:
                modules.append(("batchnorm", bn(out_channels)))
        elif char == "d":
            modules.append(("dropout", nn.Dropout(p=dropout_prob)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'")

    return modules


class SingleConv(nn.Sequential):
    """Basic convolutional module: Conv3d + non-linearity + optional norm."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        order="gcr",
        num_groups=8,
        padding=1,
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__()
        for name, module in create_conv(
            in_channels, out_channels, kernel_size, order, num_groups, padding, dropout_prob, is3d
        ):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """Two consecutive convolution layers."""

    def __init__(
        self,
        in_channels,
        out_channels,
        encoder,
        kernel_size=3,
        order="gcr",
        num_groups=8,
        padding=1,
        upscale=2,
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__()
        if encoder:
            conv1_in_channels = in_channels
            if upscale == 1:
                conv1_out_channels = out_channels
            else:
                conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        if isinstance(dropout_prob, (list, tuple)):
            dropout_prob1, dropout_prob2 = dropout_prob[0], dropout_prob[1]
        else:
            dropout_prob1 = dropout_prob2 = dropout_prob

        self.add_module(
            "SingleConv1",
            SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups, padding, dropout_prob1, is3d),
        )
        self.add_module(
            "SingleConv2",
            SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups, padding, dropout_prob2, is3d),
        )


class ResNetBlock(nn.Module):
    """Residual block for deeper UNet."""

    def __init__(self, in_channels, out_channels, kernel_size=3, order="cge", num_groups=8, is3d=True, **kwargs):
        super().__init__()

        if in_channels != out_channels:
            if is3d:
                self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
            else:
                self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv1 = nn.Identity()

        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups, is3d=is3d)
        
        n_order = order
        for c in "rel":
            n_order = n_order.replace(c, "")
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order, num_groups=num_groups, is3d=is3d)

        if "l" in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif "e" in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.conv1(x)
        out = self.conv2(residual)
        out = self.conv3(out)
        out += residual
        out = self.non_linearity(out)
        return out


class Encoder(nn.Module):
    """Encoder module: optional pooling + basic module."""

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        apply_pooling=True,
        pool_kernel_size=2,
        pool_type="max",
        basic_module=DoubleConv,
        conv_layer_order="gcr",
        num_groups=8,
        padding=1,
        upscale=2,
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__()
        assert pool_type in ["max", "avg"]
        
        if apply_pooling:
            if pool_type == "max":
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size) if is3d else nn.MaxPool2d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size) if is3d else nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(
            in_channels, out_channels, encoder=True, kernel_size=conv_kernel_size,
            order=conv_layer_order, num_groups=num_groups, padding=padding,
            upscale=upscale, dropout_prob=dropout_prob, is3d=is3d,
        )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class InterpolateUpsampling(nn.Module):
    """Non-learnable upsampling using interpolation."""

    def __init__(self, mode="nearest"):
        super().__init__()
        self.mode = mode

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return F.interpolate(x, size=output_size, mode=self.mode)


class TransposeConvUpsampling(nn.Module):
    """Learned upsampling using transposed convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2, is3d=True):
        super().__init__()
        if is3d:
            self.conv_transposed = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor, padding=1, bias=False
            )
        else:
            self.conv_transposed = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor, padding=1, bias=False
            )

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        x = self.conv_transposed(x)
        return F.interpolate(x, size=output_size)


class Decoder(nn.Module):
    """Decoder module: upsampling + joining + basic module."""

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        scale_factor=2,
        basic_module=DoubleConv,
        conv_layer_order="gcr",
        num_groups=8,
        padding=1,
        upsample="default",
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__()

        concat = True
        adapt_channels = False

        if upsample is not None and upsample != "none":
            if upsample == "default":
                if basic_module == DoubleConv:
                    upsample = "nearest"
                    concat = True
                    adapt_channels = False
                elif basic_module == ResNetBlock:
                    upsample = "deconv"
                    concat = False
                    adapt_channels = True

            if upsample == "deconv":
                self.upsampling = TransposeConvUpsampling(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=conv_kernel_size, scale_factor=scale_factor, is3d=is3d,
                )
            else:
                self.upsampling = InterpolateUpsampling(mode=upsample)
        else:
            self.upsampling = None

        self.concat = concat
        
        if adapt_channels:
            in_channels = out_channels

        self.basic_module = basic_module(
            in_channels, out_channels, encoder=False, kernel_size=conv_kernel_size,
            order=conv_layer_order, num_groups=num_groups, padding=padding,
            dropout_prob=dropout_prob, is3d=is3d,
        )

    def forward(self, encoder_features, x):
        if self.upsampling is not None:
            x = self.upsampling(encoder_features=encoder_features, x=x)
        
        if self.concat:
            x = torch.cat((encoder_features, x), dim=1)
        else:
            x = encoder_features + x
            
        x = self.basic_module(x)
        return x


def create_encoders(
    in_channels, f_maps, basic_module, conv_kernel_size, conv_padding,
    conv_upscale, dropout_prob, layer_order, num_groups, pool_kernel_size, is3d,
):
    """Create encoder path."""
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(
                in_channels, out_feature_num, apply_pooling=False,
                basic_module=basic_module, conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size, num_groups=num_groups,
                padding=conv_padding, upscale=conv_upscale,
                dropout_prob=dropout_prob, is3d=is3d,
            )
        else:
            encoder = Encoder(
                f_maps[i - 1], out_feature_num, basic_module=basic_module,
                conv_layer_order=layer_order, conv_kernel_size=conv_kernel_size,
                num_groups=num_groups, pool_kernel_size=pool_kernel_size,
                padding=conv_padding, upscale=conv_upscale,
                dropout_prob=dropout_prob, is3d=is3d,
            )
        encoders.append(encoder)
    return nn.ModuleList(encoders)


def create_decoders(
    f_maps, basic_module, conv_kernel_size, conv_padding,
    layer_order, num_groups, upsample, dropout_prob, is3d,
):
    """Create decoder path."""
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv and upsample != "deconv":
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        decoder = Decoder(
            in_feature_num, out_feature_num, basic_module=basic_module,
            conv_layer_order=layer_order, conv_kernel_size=conv_kernel_size,
            num_groups=num_groups, padding=conv_padding, upsample=upsample,
            dropout_prob=dropout_prob, is3d=is3d,
        )
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class AbstractUNet(nn.Module):
    """Base class for standard and residual UNet."""

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid,
        basic_module,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, (list, tuple))
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if "g" in layer_order:
            assert num_groups is not None

        self.encoders = create_encoders(
            in_channels, f_maps, basic_module, conv_kernel_size, conv_padding,
            conv_upscale, dropout_prob, layer_order, num_groups, pool_kernel_size, is3d,
        )

        self.decoders = create_decoders(
            f_maps, basic_module, conv_kernel_size, conv_padding,
            layer_order, num_groups, upsample, dropout_prob, is3d,
        )

        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if is_segmentation:
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = None

    def forward(self, x, return_logits=False):
        output, logits = self._forward_logits(x)
        if return_logits:
            return output, logits
        return output

    def _forward_logits(self, x):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        if self.final_activation is not None:
            out = self.final_activation(x)
            return out, x

        return x, x


class UNet3D(AbstractUNet):
    """
    3D U-Net model from "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation".
    Reference: https://arxiv.org/pdf/1606.06650.pdf
    
    Default configuration aligned with SFDA-DDFP project:
    - Channel configuration: [64, 128, 256, 512, 512] (5 levels)
    - Uses BatchNorm (layer_order="cbr") instead of GroupNorm
    - Parameter count: ~31M (comparable to DDFP 2D UNet)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=False,
        f_maps=None,  # Default to DDFP_ALIGNED_F_MAPS
        layer_order="cbr",  # Changed from "gcr" to "cbr" for BatchNorm (aligned with DDFP)
        num_groups=8,
        num_levels=5,  # Changed from 4 to 5 (aligned with DDFP)
        is_segmentation=True,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        **kwargs,
    ):
        # Use DDFP-aligned channel configuration by default
        if f_maps is None:
            f_maps = DDFP_ALIGNED_F_MAPS
        
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=DoubleConv,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=True,
        )


class ResidualUNet3D(AbstractUNet):
    """
    Residual 3D U-Net model implementation.
    Reference: https://arxiv.org/pdf/1706.00120.pdf
    
    Default configuration aligned with SFDA-DDFP project:
    - Channel configuration: [64, 128, 256, 512, 512] (5 levels)
    - Uses BatchNorm (layer_order="cbr") instead of GroupNorm
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=False,
        f_maps=None,  # Default to DDFP_ALIGNED_F_MAPS
        layer_order="cbr",  # Changed from "gcr" to "cbr" for BatchNorm (aligned with DDFP)
        num_groups=8,
        num_levels=5,
        is_segmentation=True,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        **kwargs,
    ):
        # Use DDFP-aligned channel configuration by default
        if f_maps is None:
            f_maps = DDFP_ALIGNED_F_MAPS
            
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=ResNetBlock,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=True,
        )


def get_model(model_name: str, **kwargs) -> nn.Module:
    """Factory function to get model by name."""
    models = {
        'UNet3D': UNet3D,
        'ResidualUNet3D': ResidualUNet3D,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    return models[model_name](**kwargs)
