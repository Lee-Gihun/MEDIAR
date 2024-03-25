from .custom import *

from monai.transforms import *

__all__ = [
    "train_transforms",
    "public_transforms",
    "valid_transforms",
    "tuning_transforms",
    "unlabeled_transforms",
]

train_transforms = Compose(
    [
        # >>> Load and refine data --- img: (H, W, 3); label: (H, W)
        CustomLoadImaged(keys=["img", "label"], image_only=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img", "label"], channel_dim=-1),
        RemoveRepeatedChanneld(keys=["label"], repeats=3),  # label: (H, W)
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),  # Do not scale label
        # >>> Spatial transforms
        RandZoomd(
            keys=["img", "label"],
            prob=0.5,
            min_zoom=0.25,
            max_zoom=1.5,
            mode=["area", "nearest"],
            keep_size=False,
        ),
        SpatialPadd(keys=["img", "label"], spatial_size=512),
        RandSpatialCropd(keys=["img", "label"], roi_size=512, random_size=False),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
        IntensityDiversification(keys=["img", "label"], allow_missing_keys=True),
        # # >>> Intensity transforms
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        RandGaussianSharpend(keys=["img"], prob=0.25),
        EnsureTyped(keys=["img", "label"]),
    ]
)


public_transforms = Compose(
    [
        CustomLoadImaged(keys=["img", "label"], image_only=True),
        BoundaryExclusion(keys=["label"]),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img", "label"], channel_dim=-1),
        RemoveRepeatedChanneld(keys=["label"], repeats=3),  # label: (H, W)
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),  # Do not scale label
        # >>> Spatial transforms
        SpatialPadd(keys=["img", "label"], spatial_size=512),
        RandSpatialCropd(keys=["img", "label"], roi_size=512, random_size=False),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
        Rotate90d(k=1, keys=["label"], spatial_axes=(0, 1)),
        Flipd(keys=["label"], spatial_axis=0),
        EnsureTyped(keys=["img", "label"]),
    ]
)


valid_transforms = Compose(
    [
        CustomLoadImaged(keys=["img", "label"], allow_missing_keys=True, image_only=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img", "label"], allow_missing_keys=True, channel_dim=-1),
        RemoveRepeatedChanneld(keys=["label"], repeats=3),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        EnsureTyped(keys=["img", "label"], allow_missing_keys=True),
    ]
)

tuning_transforms = Compose(
    [
        CustomLoadImaged(keys=["img"], image_only=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img"], channel_dim=-1),
        ScaleIntensityd(keys=["img"]),
        EnsureTyped(keys=["img"]),
    ]
)

unlabeled_transforms = Compose(
    [
        # >>> Load and refine data --- img: (H, W, 3); label: (H, W)
        CustomLoadImaged(keys=["img"], image_only=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img"], channel_dim=-1),
        RandZoomd(
            keys=["img"],
            prob=0.5,
            min_zoom=0.25,
            max_zoom=1.25,
            mode=["area"],
            keep_size=False,
        ),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),  # Do not scale label
        # >>> Spatial transforms
        SpatialPadd(keys=["img"], spatial_size=512),
        RandSpatialCropd(keys=["img"], roi_size=512, random_size=False),
        EnsureTyped(keys=["img"]),
    ]
)


def get_pred_transforms():
    """Prediction preprocessing"""
    pred_transforms = Compose(
        [
            # >>> Load and refine data
            CustomLoadImage(image_only=True),
            CustomNormalizeImage(channel_wise=False, percentiles=[0.0, 99.5]),
            EnsureChannelFirst(channel_dim=-1),  # image: (3, H, W)
            ScaleIntensity(),
            EnsureType(data_type="tensor"),
        ]
    )

    return pred_transforms
