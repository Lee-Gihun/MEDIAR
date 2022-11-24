import torch
import os, sys
import monai

from monai.data import decollate_batch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from core.BaseTrainer import BaseTrainer
from core.Baseline.utils import create_interior_onehot, identify_instances_from_classmap
from train_tools.measures import evaluate_f1_score_cellseg
from tqdm import tqdm

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

    def _epoch_phase(self, phase):
        """Learning process for 1 Epoch."""

        phase_results = {}

        # Set model mode
        self.model.train() if phase == "train" else self.model.eval()

        # Epoch process
        for batch_data in tqdm(self.dataloaders[phase]):
            images = batch_data["img"].to(self.device)
            labels = batch_data["label"].to(self.device)
            self.optimizer.zero_grad()

            # Map label masks to 3-class onehot map
            labels_onehot = create_interior_onehot(labels)

            # Forward pass
            with torch.set_grad_enabled(phase == "train"):
                outputs = self._inference(images, phase)
                loss = self.criterion(outputs, labels_onehot)
                self.loss_metric.append(loss)

                if phase != "train":
                    f1_score = self._get_f1_metric(outputs, labels)
                    self.f1_metric.append(f1_score)

            # Backward pass
            if phase == "train":
                # For the mixed precision training
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
            phase_results, self.loss_metric, "loss", phase
        )

        if phase != "train":
            phase_results = self._update_results(
                phase_results, self.f1_metric, "f1_score", phase
            )

        return phase_results

    def _post_process(self, outputs, labels_onehot):
        """Conduct post-processing for outputs & labels."""
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels_onehot = [self.post_gt(i) for i in decollate_batch(labels_onehot)]

        return outputs, labels_onehot

    def _get_f1_metric(self, masks_pred, masks_true):
        masks_pred = identify_instances_from_classmap(masks_pred[0])
        masks_true = masks_true.squeeze(0).squeeze(0).cpu().numpy()
        f1_score = evaluate_f1_score_cellseg(masks_true, masks_pred)[-1]

        return f1_score
