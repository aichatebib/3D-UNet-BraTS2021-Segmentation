"""MONAI preprocessing transforms for BraTS 2021."""

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    NormalizeIntensityd,
    ResizeD,
    EnsureTyped,
    MapTransform,
)

from config import SPATIAL_SIZE


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """Convert a single-channel BraTS label map into 3 binary channels.

    BraTS label convention:
        1 = necrotic / non-enhancing tumor core
        2 = peritumoral edema
        4 = enhancing tumor

    Output channels:
        0: WT (Whole Tumor)   = {1, 2, 4}
        1: TC (Tumor Core)    = {1, 4}
        2: ET (Enhancing Tumor) = {4}
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            label = d[key]

            if label.shape[0] == 1:
                label = label.squeeze(0)

            d[key] = torch.stack(
                [
                    ((label == 1) | (label == 2) | (label == 4)).float(),  # WT
                    ((label == 1) | (label == 4)).float(),                 # TC
                    (label == 4).float(),                                   # ET
                ],
                dim=0,
            )

        return d


def get_transforms(spatial_size=SPATIAL_SIZE) -> Compose:
    """Build the preprocessing pipeline shared by train/val/test sets."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ResizeD(
                keys=["image", "label"],
                spatial_size=spatial_size,
                mode=("trilinear", "nearest"),
            ),
            ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
