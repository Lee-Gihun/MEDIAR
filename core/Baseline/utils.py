"""
Adapted from the following references:
[1] https://github.com/JunMa11/NeurIPS-CellSeg/blob/main/baseline/model_training_3class.py

"""

import torch
import numpy as np
from skimage import segmentation, morphology, measure
import monai


__all__ = ["create_interior_onehot", "identify_instances_from_classmap"]


@torch.no_grad()
def identify_instances_from_classmap(
    class_map, cell_class=1, threshold=0.5, from_logits=True
):
    """Identification of cell instances from the class map"""

    if from_logits:
        class_map = torch.softmax(class_map, dim=0)  # (C, H, W)

    # Convert probability map to binary mask
    pred_mask = class_map[cell_class].cpu().numpy()

    # Apply morphological postprocessing
    pred_mask = pred_mask > threshold
    pred_mask = morphology.remove_small_holes(pred_mask, connectivity=1)
    pred_mask = morphology.remove_small_objects(pred_mask, 16)
    pred_mask = measure.label(pred_mask)

    return pred_mask


@torch.no_grad()
def create_interior_onehot(inst_maps):
    """
    interior : (H,W), np.uint8
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    device = inst_maps.device

    # Get (np.int16) array corresponding to label masks: (B, 1, H, W)
    inst_maps = inst_maps.squeeze(1).cpu().numpy().astype(np.int16)

    interior_maps = []

    for inst_map in inst_maps:
        # Create interior-edge map
        boundary = segmentation.find_boundaries(inst_map, mode="inner")

        # Refine interior-edge map
        boundary = morphology.binary_dilation(boundary, morphology.disk(1))

        # Assign label classes
        interior_temp = np.logical_and(~boundary, inst_map > 0)

        # interior_temp[boundary] = 0
        interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
        interior = np.zeros_like(inst_map, dtype=np.uint8)
        interior[interior_temp] = 1
        interior[boundary] = 2

        interior_maps.append(interior)

    # Aggregate interior_maps for batch
    interior_maps = np.stack(interior_maps, axis=0).astype(np.uint8)

    # Reshape as original label shape: (B, H, W)
    interior_maps = torch.from_numpy(interior_maps).unsqueeze(1).to(device)

    # Obtain one-hot map for batch
    interior_onehot = monai.networks.one_hot(interior_maps, num_classes=3)

    return interior_onehot
