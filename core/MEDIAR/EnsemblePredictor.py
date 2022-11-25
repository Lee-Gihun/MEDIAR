import torch
import os, sys, copy
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from core.MEDIAR.Predictor import Predictor

__all__ = ["EnsemblePredictor"]


class EnsemblePredictor(Predictor):
    def __init__(
        self,
        model,
        model_aux,
        device,
        input_path,
        output_path,
        make_submission=False,
        exp_name=None,
        algo_params=None,
    ):
        super(EnsemblePredictor, self).__init__(
            model,
            device,
            input_path,
            output_path,
            make_submission,
            exp_name,
            algo_params,
        )
        self.model_aux = model_aux

    @torch.no_grad()
    def _inference(self, img_data):

        self.model_aux.to(self.device)
        self.model_aux.eval()

        img_data = img_data.to(self.device)
        img_base = img_data

        outputs_base = self._window_inference(img_base)
        outputs_base = outputs_base.cpu().squeeze()

        outputs_aux = self._window_inference(img_base, aux=True)
        outputs_aux = outputs_aux.cpu().squeeze()
        img_base.cpu()

        if not self.use_tta:
            pred_mask = (outputs_base + outputs_aux) / 2
            return pred_mask

        else:
            # HorizontalFlip TTA
            img_hflip = self.hflip_tta.apply_aug_image(img_data, apply=True)

            outputs_hflip = self._window_inference(img_hflip)
            outputs_hflip_aux = self._window_inference(img_hflip, aux=True)

            outputs_hflip = self.hflip_tta.apply_deaug_mask(outputs_hflip, apply=True)
            outputs_hflip_aux = self.hflip_tta.apply_deaug_mask(
                outputs_hflip_aux, apply=True
            )

            outputs_hflip = outputs_hflip.cpu().squeeze()
            outputs_hflip_aux = outputs_hflip_aux.cpu().squeeze()
            img_hflip = img_hflip.cpu()

            # VertricalFlip TTA
            img_vflip = self.vflip_tta.apply_aug_image(img_data, apply=True)

            outputs_vflip = self._window_inference(img_vflip)
            outputs_vflip_aux = self._window_inference(img_vflip, aux=True)

            outputs_vflip = self.vflip_tta.apply_deaug_mask(outputs_vflip, apply=True)
            outputs_vflip_aux = self.vflip_tta.apply_deaug_mask(
                outputs_vflip_aux, apply=True
            )

            outputs_vflip = outputs_vflip.cpu().squeeze()
            outputs_vflip_aux = outputs_vflip_aux.cpu().squeeze()
            img_vflip = img_vflip.cpu()

            # Merge Results
            pred_mask = torch.zeros_like(outputs_base)
            pred_mask[0] = (outputs_base[0] + outputs_hflip[0] - outputs_vflip[0]) / 3
            pred_mask[1] = (outputs_base[1] - outputs_hflip[1] + outputs_vflip[1]) / 3
            pred_mask[2] = (outputs_base[2] + outputs_hflip[2] + outputs_vflip[2]) / 3

            pred_mask_aux = torch.zeros_like(outputs_aux)
            pred_mask_aux[0] = (
                outputs_aux[0] + outputs_hflip_aux[0] - outputs_vflip_aux[0]
            ) / 3
            pred_mask_aux[1] = (
                outputs_aux[1] - outputs_hflip_aux[1] + outputs_vflip_aux[1]
            ) / 3
            pred_mask_aux[2] = (
                outputs_aux[2] + outputs_hflip_aux[2] + outputs_vflip_aux[2]
            ) / 3

            pred_mask = (pred_mask + pred_mask_aux) / 2

        return pred_mask
