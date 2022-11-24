import torch
import numpy as np
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.metrics import CumulativeAverage
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)

import os, sys
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from core.utils import print_learning_device, print_with_logging
from train_tools.measures import evaluate_f1_score_cellseg


class BaseTrainer:
    """Abstract base class for trainer implementations"""

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
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.no_valid = no_valid
        self.valid_frequency = valid_frequency
        self.device = device
        self.amp = amp
        self.best_weights = None
        self.best_f1_score = 0.1

        # FP-16 Scaler
        self.scaler = torch.cuda.amp.GradScaler() if amp else None

        # Assign algoritm-specific arguments
        if algo_params:
            self.__dict__.update((k, v) for k, v in algo_params.items())

        # Cumulitive statistics
        self.loss_metric = CumulativeAverage()
        self.f1_metric = CumulativeAverage()

        # Post-processing functions
        self.post_pred = Compose(
            [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
        )
        self.post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])

    def train(self):
        """Train the model"""

        # Print learning device name
        print_learning_device(self.device)

        # Learning process
        for epoch in range(1, self.num_epochs + 1):
            print(f"[Round {epoch}/{self.num_epochs}]")

            # Train Epoch Phase
            print(">>> Train Epoch")
            train_results = self._epoch_phase("train")
            print_with_logging(train_results, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            if epoch % self.valid_frequency == 0:
                if not self.no_valid:
                    # Valid Epoch Phase
                    print(">>> Valid Epoch")
                    valid_results = self._epoch_phase("valid")
                    print_with_logging(valid_results, epoch)

                    if "Valid_F1_Score" in valid_results.keys():
                        current_f1_score = valid_results["Valid_F1_Score"]
                        self._update_best_model(current_f1_score)
                else:
                    print(">>> TuningSet Epoch")
                    tuning_cell_counts = self._tuningset_evaluation()
                    tuning_count_dict = {"TuningSet_Cell_Count": tuning_cell_counts}
                    print_with_logging(tuning_count_dict, epoch)

                    current_cell_count = tuning_cell_counts
                    self._update_best_model(current_cell_count)

            print("-" * 50)

            self.best_f1_score = 0

        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)

    def _epoch_phase(self, phase):
        """Learning process for 1 Epoch (for different phases).

        Args:
            phase (str): "train", "valid", "test"

        Returns:
            dict: statistics for the phase results
        """
        phase_results = {}

        # Set model mode
        self.model.train() if phase == "train" else self.model.eval()

        # Epoch process
        for batch_data in tqdm(self.dataloaders[phase]):
            images = batch_data["img"].to(self.device)
            labels = batch_data["label"].to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(phase == "train"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.loss_metric.append(loss)

            # Backward pass
            if phase == "train":
                loss.backward()
                self.optimizer.step()

        # Update metrics
        phase_results = self._update_results(
            phase_results, self.loss_metric, "loss", phase
        )

        return phase_results

    @torch.no_grad()
    def _tuningset_evaluation(self):
        cell_counts_total = []
        self.model.eval()

        for batch_data in tqdm(self.dataloaders["tuning"]):
            images = batch_data["img"].to(self.device)
            if images.shape[-1] > 5000:
                continue

            outputs = sliding_window_inference(
                images,
                roi_size=512,
                sw_batch_size=4,
                predictor=self.model,
                padding_mode="constant",
                mode="gaussian",
            )

            outputs = outputs.squeeze(0)
            outputs, _ = self._post_process(outputs, None)
            count = len(np.unique(outputs) - 1)
            cell_counts_total.append(count)

        cell_counts_total_sum = np.sum(cell_counts_total)
        print("Cell Counts Total: (%d)" % (cell_counts_total_sum))

        return cell_counts_total_sum

    def _update_results(self, phase_results, metric, metric_key, phase="train"):
        """Aggregate and flush metrics

        Args:
            phase_results (dict): base dictionary to log metrics
            metric (_type_): cumulated metrics
            metric_key (_type_): name of metric
            phase (str, optional): current phase name. Defaults to "train".

        Returns:
            dict: dictionary of metrics for the current phase
        """

        # Refine metrics name
        metric_key = "_".join([phase, metric_key]).title()

        # Aggregate metrics
        metric_item = round(metric.aggregate().item(), 4)

        # Log metrics to dictionary
        phase_results[metric_key] = metric_item

        # Flush metrics
        metric.reset()

        return phase_results

    def _update_best_model(self, current_f1_score):
        if current_f1_score > self.best_f1_score:
            self.best_weights = copy.deepcopy(self.model.state_dict())
            self.best_f1_score = current_f1_score
            print(
                "\n>>>> Update Best Model with score: {}\n".format(self.best_f1_score)
            )
        else:
            pass

    def _inference(self, images, phase="train"):
        """inference methods for different phase"""
        if phase != "train":
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

    def _post_process(self, outputs, labels):
        return outputs, labels

    def _get_f1_metric(self, masks_pred, masks_true):
        f1_score = evaluate_f1_score_cellseg(masks_true, masks_pred)[-1]

        return f1_score
