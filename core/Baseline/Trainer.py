import os
import sys
import monai
from monai.inferers import sliding_window_inference
from monai.metrics import CumulativeAverage, DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)
from monai.data import decollate_batch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from core.BaseTrainer import BaseTrainer
from core.Baseline.utils import create_interior_onehot, identify_instances_from_classmap
from train_tools.measures import evaluate_f1_score_cellseg

__all__ = ["Trainer"]


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        dataloaders,
        optimizer,
        scheduler=None,
        criterion=None,
        num_epochs=100,
        device="cuda:0",
        no_valid=False,
        valid_frequency=1,
        amp=False,
        algo_params=None,
    ):
        super(Trainer, self).__init__(
            model,
            dataloaders,
            optimizer,
            scheduler,
            criterion,
            num_epochs,
            device,
            no_valid,
            valid_frequency,
            amp,
            algo_params,
        )

        # Dice loss as segmentation criterion
        self.criterion = monai.losses.DiceCELoss(softmax=True)

        # Post-processing functions
        self.post_pred = Compose(
            [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
        )
        self.post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])

        # Cumulitive statistics
        self.loss_metric = CumulativeAverage()
        self.f1_metric = CumulativeAverage()
        self.score_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )

    def _eval_instance(self, masks_pred, masks_true):
        masks_pred = identify_instances_from_classmap(masks_pred)
        _, _, f1_score = evaluate_f1_score_cellseg(
            masks_true=masks_true.squeeze(0).squeeze(0).cpu().numpy(),
            masks_pred=masks_pred,
        )
        return f1_score

    def _inference(self, images, phase="train"):
        """inference methods for different phase"""
        if phase == "valid":
            outputs = sliding_window_inference(
                images,
                roi_size=512,
                sw_batch_size=4,
                predictor=self.model,
                padding_mode="reflect",
                mode="gaussian",
                overlap=0.5,
            )
        else:
            outputs = self.model(images)

        return outputs

    def _post_process(self, outputs, labels_onehot):
        """Conduct post-processing for outputs & labels."""
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels_onehot = [self.post_gt(i) for i in decollate_batch(labels_onehot)]

        return outputs, labels_onehot
