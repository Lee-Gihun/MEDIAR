import numpy as np
from skimage import exposure
from monai.config import KeysCollection

from monai.transforms.transform import Transform
from monai.transforms.compose import MapTransform

from typing import Dict, Hashable, Mapping


__all__ = [
    "CustomNormalizeImage",
    "CustomNormalizeImageD",
    "CustomNormalizeImageDict",
    "CustomNormalizeImaged",
]


class CustomNormalizeImage(Transform):
    """Normalize the image."""

    def __init__(self, percentiles=[0, 99.5], channel_wise=False):
        self.lower, self.upper = percentiles
        self.channel_wise = channel_wise

    def _normalize(self, img) -> np.ndarray:
        non_zero_vals = img[np.nonzero(img)]
        percentiles = np.percentile(non_zero_vals, [self.lower, self.upper])
        img_norm = exposure.rescale_intensity(
            img, in_range=(percentiles[0], percentiles[1]), out_range="uint8"
        )

        return img_norm.astype(np.uint8)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.channel_wise:
            pre_img_data = np.zeros(img.shape, dtype=np.uint8)
            for i in range(img.shape[-1]):
                img_channel_i = img[:, :, i]

                if len(img_channel_i[np.nonzero(img_channel_i)]) > 0:
                    pre_img_data[:, :, i] = self._normalize(img_channel_i)

            img = pre_img_data

        else:
            img = self._normalize(img)

        return img


class CustomNormalizeImaged(MapTransform):
    """Dictionary-based wrapper of NormalizeImage"""

    def __init__(
        self,
        keys: KeysCollection,
        percentiles=[1, 99],
        channel_wise: bool = False,
        allow_missing_keys: bool = False,
    ):
        super(CustomNormalizeImageD, self).__init__(keys, allow_missing_keys)
        self.normalizer = CustomNormalizeImage(percentiles, channel_wise)

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:

        d = dict(data)

        for key in self.keys:
            d[key] = self.normalizer(d[key])

        return d


CustomNormalizeImageD = CustomNormalizeImageDict = CustomNormalizeImaged
