"""3D U-Net model definition (MONAI)."""

from monai.networks.nets import UNet

from config import (
    IN_CHANNELS,
    OUT_CHANNELS,
    CHANNELS,
    STRIDES,
    NUM_RES_UNITS,
    DEVICE,
)


def build_model():
    """Build the 3D multi-class U-Net used for WT/TC/ET segmentation."""
    model = UNet(
        spatial_dims=3,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        channels=CHANNELS,
        strides=STRIDES,
        num_res_units=NUM_RES_UNITS,
    ).to(DEVICE)

    return model
