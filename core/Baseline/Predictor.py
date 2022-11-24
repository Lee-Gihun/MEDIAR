import torch
import os, sys
from skimage import morphology, measure
from monai.inferers import sliding_window_inference

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from core.BasePredictor import BasePredictor

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

    def _inference(self, img_data):
        pred_mask = sliding_window_inference(
            img_data,
            512,
            4,
            self.model,
            padding_mode="constant",
            mode="gaussian",
            overlap=0.6,
        )

        return pred_mask

    def _post_process(self, pred_mask):
        # Get probability map from the predicted logits
        pred_mask = torch.from_numpy(pred_mask)
        pred_mask = torch.softmax(pred_mask, dim=0)
        pred_mask = pred_mask[1].cpu().numpy()

        # Apply morphological post-processing
        pred_mask = pred_mask > 0.5
        pred_mask = morphology.remove_small_holes(pred_mask, connectivity=1)
        pred_mask = morphology.remove_small_objects(pred_mask, 16)
        pred_mask = measure.label(pred_mask)

        return pred_mask
