import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.metrics import CumulativeAverage, DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from core.BaseTrainer import BaseTrainer
from core.Cellpose.utils import *
from train_tools.data_utils.custom.cellseg_mix import *

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
        # originally cellpose use binary classification
        # mse loss for gradients
        self.criterion1 = nn.MSELoss(reduction="mean")
        self.criterion2 = nn.BCEWithLogitsLoss(reduction="mean")

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

    def loss_fn(self, lbl, y, device):
        """loss function between true labels lbl and prediction y"""

        veci = 5.0 * torch.from_numpy(lbl[:, 2:]).to(device)
        loss = self.criterion1(y[:, :2], veci)
        loss /= 2.0
        loss2 = self.criterion2(
            y[:, -1], torch.from_numpy(lbl[:, 1] > 0.5).to(device).float()
        )
        loss = loss + loss2
        return loss

    def _epoch_phase(self, phase):
        phase_results = {}

        # Set model mode
        self.model.train() if phase == "train" else self.model.eval()

        # Epoch process
        for batch_data in tqdm(self.dataloaders[phase]):
            images, labels = batch_data["img"], batch_data["label"]

            if self.with_public:
                # Load batches sequentially from the unlabeled dataloader
                try:
                    batch_data = next(self.public_iterator)
                    images_pub, labels_pub = batch_data["img"], batch_data["label"]

                except:
                    self.public_iterator = iter(self.public_loader)
                    batch_data = next(self.public_iterator)
                    images_pub, labels_pub = batch_data["img"], batch_data["label"]

                images = torch.cat([images, images_pub], dim=0)
                labels = torch.cat([labels, labels_pub], dim=0)

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Map label masks to graidnet and onehot
            labels_onehot_flows = labels_to_flows(
                labels, use_gpu=True, device=self.device
            )

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.amp):
                with torch.set_grad_enabled(phase == "train"):
                    # outputs are B x [grad y, grad x, cellprob] x H x W
                    outputs = self._inference(images, phase)

                    # Calculate & dice loss and gradient loss
                    loss = self.loss_fn(labels_onehot_flows, outputs, self.device)
                    self.loss_metric.append(loss)

                    if phase != "train":
                        outputs, labels = self._post_process(outputs, labels)
                        f1_score = self._get_f1_metric(outputs, labels)
                        self.f1_metric.append(f1_score)

                # Backward pass
                if phase == "train":
                    if self.amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    else:
                        loss.backward()
                        self.optimizer.step()

        # Update metrics
        phase_results = self._update_results(
            phase_results, self.loss_metric, "dice_loss", phase
        )
        if phase != "train":
            phase_results = self._update_results(
                phase_results, self.f1_metric, "f1_score", phase
            )

        return phase_results

    def _inference(self, images, phase="train"):
        """inference methods for different phase"""
        if phase != "train":
            outputs = sliding_window_inference(
                images,
                roi_size=512,
                sw_batch_size=4,
                predictor=self.model,
                padding_mode="constant",
                mode="gaussian",
                overlap=0.6,
            )
        else:
            outputs = self.model(images)

        return outputs

    def _post_process(self, outputs, labels=None):
        outputs = outputs.squeeze(0).cpu().numpy()
        dP, cellprob = outputs[:2], self._sigmoid(outputs[-1])
        outputs = compute_masks(dP, cellprob, use_gpu=True, device=self.device)[0]

        if labels is not None:
            labels = labels.squeeze(0).squeeze(0).cpu().numpy()

        return outputs, labels

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
