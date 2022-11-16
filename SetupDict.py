import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import monai
import core
from train_tools import models
from train_tools.models import *

__all__ = ["PATHS", "TRAINER", "OPTIMIZER", "SCHEDULER"]

# [!Caution] The paths should be overrided for the local environment!
PATHS = {
    "root": "/home/gihun/data/CellSeg/",
    "train_labeled_images": "Official/Train_Labeled/images/",
    "train_labeled_labels": "Official/Train_Labeled/labels/",
    "train_unlabeled_images": "Official/Train_Unlabeled/images/",
    "tuning_images": "Official/TuningSet",
}

TRAINER = {
    "baseline": core.Baseline.Trainer,
    "mediar": core.MEDIAR.Trainer,
}

PREDICTOR = {
    "baseline": core.Baseline.Predictor,
    "mediar": core.MEDIAR.PredictorTTA,
}

MODELS = {
    "unet": monai.networks.nets.UNet,
    "unetr": monai.networks.nets.unetr.UNETR,
    "swinunetr": monai.networks.nets.SwinUNETR,
    "mediar-former": models.MEDIARFormer,
}

OPTIMIZER = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
}

SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}
