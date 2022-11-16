import numpy as np
import copy

from monai.transforms import RandScaleIntensity, Compose
from monai.transforms.compose import MapTransform
from skimage.segmentation import find_boundaries


__all__ = ["BoundaryExclusion", "IntensityDiversification"]


class BoundaryExclusion(MapTransform):
    """Map the cell boundary pixel labels to the background class (0)."""

    def __init__(self, keys=["label"], allow_missing_keys=False):
        super(BoundaryExclusion, self).__init__(keys, allow_missing_keys)

    def __call__(self, data):
        # Find and Exclude Boundary
        label_original = data["label"]
        label = copy.deepcopy(label_original)
        boundary = find_boundaries(label, connectivity=1, mode="thick")
        label[boundary] = 0

        # Do not exclude if the cell is too small (< 14x14).
        new_label = copy.deepcopy(label_original)
        new_label[label == 0] = 0

        cell_idx, cell_counts = np.unique(label_original, return_counts=True)

        for k in range(len(cell_counts)):
            if cell_counts[k] < 196:
                new_label[label_original == cell_idx[k]] = cell_idx[k]

        # Do not exclude if the pixels are at the image boundaries.
        _, W, H = label_original.shape
        bd = np.zeros_like(label_original, dtype=label.dtype)
        bd[:, 2 : W - 2, 2 : H - 2] = 1
        new_label += label_original * bd

        # Assign the transformed label
        data["label"] = new_label

        return data


class IntensityDiversification(MapTransform):
    """Randomly rescale the intensity of cell pixels."""

    def __init__(
        self,
        keys=["img"],
        change_cell_ratio=0.4,
        scale_factors=[0, 0.7],
        allow_missing_keys=False,
    ):
        super(IntensityDiversification, self).__init__(keys, allow_missing_keys)

        self.change_cell_ratio = change_cell_ratio
        self.randscale_intensity = Compose(
            [RandScaleIntensity(prob=1.0, factors=scale_factors)]
        )

    def __call__(self, data):
        # Select cells to be transformed
        cell_count = int(data["label"].max())
        change_cell_count = int(cell_count * self.change_cell_ratio)
        change_cell = np.random.choice(cell_count, change_cell_count, replace=False)

        mask = copy.deepcopy(data["label"])

        for i in range(cell_count):
            cell_id = i + 1

            if cell_id not in change_cell:
                mask[mask == cell_id] = 0

        mask[mask > 0] = 1

        # Conduct intensity transformation for the selected cells
        img_original = copy.deepcopy((1 - mask) * data["img"])
        img_transformed = copy.deepcopy(mask * data["img"])
        img_transformed = self.randscale_intensity(img_transformed)

        # Assign the transformed image
        data["img"] = img_original + img_transformed

        return data
