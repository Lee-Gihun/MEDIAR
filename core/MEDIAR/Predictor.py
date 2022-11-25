import torch
import numpy as np
import os, sys
from monai.inferers import sliding_window_inference

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from core.BasePredictor import BasePredictor
from core.MEDIAR.utils import compute_masks

__all__ = ["Predictor"]


class Predictor(BasePredictor):
    def __init__(
        self,
        model,
        device,
        input_path,
        output_path,
        make_submission=False,
        exp_name=None,
        algo_params=None,
    ):
        super(Predictor, self).__init__(
            model,
            device,
            input_path,
            output_path,
            make_submission,
            exp_name,
            algo_params,
        )
        self.hflip_tta = HorizontalFlip()
        self.vflip_tta = VerticalFlip()

    @torch.no_grad()
    def _inference(self, img_data):
        """Conduct model prediction"""

        img_data = img_data.to(self.device)
        img_base = img_data
        outputs_base = self._window_inference(img_base)
        outputs_base = outputs_base.cpu().squeeze()
        img_base.cpu()

        if not self.use_tta:
            pred_mask = outputs_base
            return pred_mask

        else:
            # HorizontalFlip TTA
            img_hflip = self.hflip_tta.apply_aug_image(img_data, apply=True)
            outputs_hflip = self._window_inference(img_hflip)
            outputs_hflip = self.hflip_tta.apply_deaug_mask(outputs_hflip, apply=True)
            outputs_hflip = outputs_hflip.cpu().squeeze()
            img_hflip = img_hflip.cpu()

            # VertricalFlip TTA
            img_vflip = self.vflip_tta.apply_aug_image(img_data, apply=True)
            outputs_vflip = self._window_inference(img_vflip)
            outputs_vflip = self.vflip_tta.apply_deaug_mask(outputs_vflip, apply=True)
            outputs_vflip = outputs_vflip.cpu().squeeze()
            img_vflip = img_vflip.cpu()

            # Merge Results
            pred_mask = torch.zeros_like(outputs_base)
            pred_mask[0] = (outputs_base[0] + outputs_hflip[0] - outputs_vflip[0]) / 3
            pred_mask[1] = (outputs_base[1] - outputs_hflip[1] + outputs_vflip[1]) / 3
            pred_mask[2] = (outputs_base[2] + outputs_hflip[2] + outputs_vflip[2]) / 3

        return pred_mask

    def _window_inference(self, img_data, aux=False):
        """Inference on RoI-sized window"""
        outputs = sliding_window_inference(
            img_data,
            roi_size=512,
            sw_batch_size=4,
            predictor=self.model if not aux else self.model_aux,
            padding_mode="constant",
            mode="gaussian",
            overlap=0.6,
        )

        return outputs

    def _post_process(self, pred_mask):
        """Generate cell instance masks."""
        dP, cellprob = pred_mask[:2], self._sigmoid(pred_mask[-1])
        H, W = pred_mask.shape[-2], pred_mask.shape[-1]

        if np.prod(H * W) < (5000 * 5000):
            pred_mask = compute_masks(
                dP,
                cellprob,
                use_gpu=True,
                flow_threshold=0.4,
                device=self.device,
                cellprob_threshold=0.5,
            )[0]

        else:
            print("\n[Whole Slide] Grid Prediction starting...")
            roi_size = 2000

            # Get patch grid by roi_size
            if H % roi_size != 0:
                n_H = H // roi_size + 1
                new_H = roi_size * n_H
            else:
                n_H = H // roi_size
                new_H = H

            if W % roi_size != 0:
                n_W = W // roi_size + 1
                new_W = roi_size * n_W
            else:
                n_W = W // roi_size
                new_W = W

            # Allocate values on the grid
            pred_pad = np.zeros((new_H, new_W), dtype=np.uint32)
            dP_pad = np.zeros((2, new_H, new_W), dtype=np.float32)
            cellprob_pad = np.zeros((new_H, new_W), dtype=np.float32)

            dP_pad[:, :H, :W], cellprob_pad[:H, :W] = dP, cellprob

            for i in range(n_H):
                for j in range(n_W):
                    print("Pred on Grid (%d, %d) processing..." % (i, j))
                    dP_roi = dP_pad[
                        :,
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ]
                    cellprob_roi = cellprob_pad[
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ]

                    pred_mask = compute_masks(
                        dP_roi,
                        cellprob_roi,
                        use_gpu=True,
                        flow_threshold=0.4,
                        device=self.device,
                        cellprob_threshold=0.5,
                    )[0]

                    pred_pad[
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ] = pred_mask

            pred_mask = pred_pad[:H, :W]

        return pred_mask

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


"""
Adapted from the following references:
[1] https://github.com/qubvel/ttach/blob/master/ttach/transforms.py

"""


def hflip(x):
    """flip batch of images horizontally"""
    return x.flip(3)


def vflip(x):
    """flip batch of images vertically"""
    return x.flip(2)


class DualTransform:
    identity_param = None

    def __init__(
        self, name: str, params,
    ):
        self.params = params
        self.pname = name

    def apply_aug_image(self, image, *args, **params):
        raise NotImplementedError

    def apply_deaug_mask(self, mask, *args, **params):
        raise NotImplementedError


class HorizontalFlip(DualTransform):
    """Flip images horizontally (left -> right)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = hflip(image)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = hflip(mask)
        return mask


class VerticalFlip(DualTransform):
    """Flip images vertically (up -> down)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = vflip(image)

        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = vflip(mask)

        return mask
