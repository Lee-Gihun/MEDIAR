import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.data import Dataset, CacheDataset

from .transforms import *
from .utils import *

import pickle

LABELED_JSON_FILE = "./mapping_labeled.json"
UNLABELED_JSON_FILE = "./mapping_unlabeled.json"
TUNING_JSON_FILE = "./mapping_tuning.json"
PUBLIC_JSON_FILE = "./mapping_public.json"

DATA_LABEL_DICT_PICKLE_FILE = "./train_tools/data_utils/custom/data_dict_ver3.pickle"

__all__ = [
    "get_dataloaders_joint",
    "get_dataloaders_labeled",
    "get_dataloaders_unlabeled",
]


def get_dataloaders_joint(
    root,
    mapping_file_official,
    mapping_file_public,
    mapping_file_tuning="/home/gihun/CellSeg/train_tools/data_utils/mapping_tuning.json",
    map_keys="all",
    valid_portion=0.0,
    batch_size=8,
    num_workers=5,
    data_aug=False,
    relabel=False,
    use_cache=False,
):
    # Get list of data dictionaries from decoded paths
    data_dicts = path_decoder(root, mapping_file_official, map_keys)
    tuning_dicts = path_decoder(root, mapping_file_tuning, no_label=True)

    if data_aug:
        with open(DATA_LABEL_DICT_PICKLE_FILE, "rb") as f:
            data_label_dict = pickle.load(f)

        data_point_dict = {}

        for label, data_lst in data_label_dict.items():
            data_point_dict[label] = []

            for d_idx in data_lst:
                try:
                    data_point_dict[label].append(data_dicts[d_idx])
                except:
                    print(label, d_idx)

        data_dicts = []

        for label, data_points in data_point_dict.items():
            len_data_points = len(data_points)

            if len_data_points >= 50:
                data_dicts += data_points
            else:
                if label == 7:
                    for i in range(126):
                        data_dicts.append(data_points[i % len_data_points])
                else:
                    for i in range(50):
                        data_dicts.append(data_points[i % len_data_points])

    if relabel:
        for elem in data_dicts:
            cell_idx = int(elem["label"].split("_label.tiff")[0].split("_")[-1])
            if cell_idx in range(340, 499):
                new_label = elem["label"].replace(
                    "/data/CellSeg/Official/Train_Labeled/labels/",
                    "/CellSeg/pretrained_train_ext/",
                )
                elem["label"] = new_label

    public_dicts = path_decoder(root, mapping_file_public, map_keys)

    for elem in public_dicts:
        ex_label = elem["label"].replace("labels", "ex_labels")
        elem["label"] = ex_label
        data_dicts.append(elem)

    # Use caching for non-random transforms when use_cache
    DATACLASS = CacheDataset if use_cache else Dataset

    # Split datasets as Train/Valid
    train_dicts, valid_dicts = split_train_valid(
        data_dicts, valid_portion=valid_portion
    )

    # Obtain datasets with transforms
    trainset = DATACLASS(train_dicts, transform=train_transforms)
    validset = DATACLASS(valid_dicts, transform=valid_transforms)
    tuningset = DATACLASS(tuning_dicts, transform=tuning_transforms)

    # Set dataloader for Trainset
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # Set dataloader for Validset (Batch size is fixed as 1)
    valid_loader = DataLoader(validset, batch_size=1, shuffle=False,)

    # Set dataloader for Tuningset (Batch size is fixed as 1)
    tuning_loader = DataLoader(tuningset, batch_size=1, shuffle=False)

    # Form dataloaders as dictionary
    dataloaders = {
        "train": train_loader,
        "valid": valid_loader,
        "tuning": tuning_loader,
    }

    return dataloaders


def get_dataloaders_labeled(
    root,
    mapping_file,
    mapping_file_tuning="/home/gihun/CellSeg/train_tools/data_utils/mapping_tuning.json",
    map_keys="all",
    valid_portion=0.1,
    batch_size=8,
    num_workers=5,
    use_cache=False,
    data_aug=False,
    relabel=False,
):
    """Set DataLoaders for labeled datasets.

    Args:
        root (str): root directory
        mapping_file (str): json file for mapping dataset
        map_keys (list or str, optional): using dataset names. Defaults to "all".
        valid_portion (float, optional): portion of valid datasets. Defaults to 0.1.
        batch_size (int, optional): batch size. Defaults to 8.
        shuffle (bool, optional): shuffles dataloader. Defaults to True.
        num_workers (int, optional): number of workers for each datalaoder. Defaults to 5.
        use_cache (bool, optional): wheter to use caching for non-random transforms. Defaults to False.

    Returns:
        dict: dictionary of data loaders.
    """

    # Get list of data dictionaries from decoded paths
    data_dicts = path_decoder(root, mapping_file, map_keys)
    tuning_dicts = path_decoder(root, mapping_file_tuning, no_label=True)

    if data_aug:
        with open(DATA_LABEL_DICT_PICKLE_FILE, "rb") as f:
            data_label_dict = pickle.load(f)

        data_point_dict = {}

        for label, data_lst in data_label_dict.items():
            data_point_dict[label] = []

            for d_idx in data_lst:
                try:
                    data_point_dict[label].append(data_dicts[d_idx])
                except:
                    print(label, d_idx)

        data_dicts = []

        for label, data_points in data_point_dict.items():
            len_data_points = len(data_points)

            if len_data_points >= 50:
                data_dicts += data_points
            else:
                if label == 7:
                    for i in range(126):
                        data_dicts.append(data_points[i % len_data_points])
                else:
                    for i in range(50):
                        data_dicts.append(data_points[i % len_data_points])

    if relabel:
        for elem in data_dicts:
            cell_idx = int(elem["label"].split("_label.tiff")[0].split("_")[-1])
            if cell_idx in range(340, 499):
                new_label = elem["label"].replace(
                    "/data/CellSeg/Official/Train_Labeled/labels/",
                    "/CellSeg/pretrained_train_ext/",
                )
                elem["label"] = new_label

    # Use caching for non-random transforms when use_cache
    DATACLASS = CacheDataset if use_cache else Dataset

    # Split datasets as Train/Valid
    train_dicts, valid_dicts = split_train_valid(
        data_dicts, valid_portion=valid_portion
    )

    # Obtain datasets with transforms
    trainset = DATACLASS(train_dicts, transform=train_transforms)
    validset = DATACLASS(valid_dicts, transform=valid_transforms)
    tuningset = DATACLASS(tuning_dicts, transform=tuning_transforms)

    # Set dataloader for Trainset
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # Set dataloader for Validset (Batch size is fixed as 1)
    valid_loader = DataLoader(validset, batch_size=1, shuffle=False,)

    # Set dataloader for Tuningset (Batch size is fixed as 1)
    tuning_loader = DataLoader(tuningset, batch_size=1, shuffle=False)

    # Form dataloaders as dictionary
    dataloaders = {
        "train": train_loader,
        "valid": valid_loader,
        "tuning": tuning_loader,
    }

    return dataloaders


def get_dataloaders_public(
    root,
    mapping_file,
    map_keys="all",
    valid_portion=0.0,
    batch_size=8,
    num_workers=5,
    use_train_transforms=False,
    partial=False,
    use_cache=False,
):
    """Set DataLoaders for labeled datasets.

    Args:
        root (str): root directory
        mapping_file (str): json file for mapping dataset
        map_keys (list or str, optional): using dataset names. Defaults to "all".
        valid_portion (float, optional): portion of valid datasets. Defaults to 0.1.
        batch_size (int, optional): batch size. Defaults to 8.
        shuffle (bool, optional): shuffles dataloader. Defaults to True.
        num_workers (int, optional): number of workers for each datalaoder. Defaults to 5.
        use_cache (bool, optional): wheter to use caching for non-random transforms. Defaults to False.

    Returns:
        dict: dictionary of data loaders.
    """

    # Get list of data dictionaries from decoded paths
    data_dicts = path_decoder(root, mapping_file, map_keys)

    # Use caching for non-random transforms when use_cache
    DATACLASS = CacheDataset if use_cache else Dataset

    # Split datasets as Train/Valid
    train_dicts, valid_dicts = split_train_valid(
        data_dicts, valid_portion=valid_portion
    )

    if partial:
        train_dicts = train_dicts[:615]

    # Obtain datasets with transforms
    if use_train_transforms:
        transforms = train_transforms
    else:
        transforms = public_transforms

    trainset = DATACLASS(train_dicts, transform=transforms)
    # Set dataloader for Trainset
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # Form dataloaders as dictionary
    dataloaders = {
        "public": train_loader,
    }

    return dataloaders


def get_dataloaders_unlabeled(
    root,
    mapping_file=UNLABELED_JSON_FILE,
    map_keys="all",
    batch_size=8,
    shuffle=True,
    num_workers=5,
    use_cache=False,
):
    """Set dataloaders for unlabeled dataset."""
    # Get list of data dictionaries from decoded paths
    unlabeled_dicts = path_decoder(
        root, mapping_file, map_keys, no_label=True, unlabeled=True
    )

    # Use caching for non-random transforms when use_cache
    DATACLASS = CacheDataset if use_cache else Dataset

    # Obtain datasets with transforms
    unlabeled_dicts, _ = split_train_valid(unlabeled_dicts, valid_portion=0)
    unlabeled_set = DATACLASS(unlabeled_dicts, transform=unlabeled_transforms)

    # Set dataloader for Unlabeled dataset
    unlabeled_loader = DataLoader(
        unlabeled_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    dataloaders = {
        "unlabeled": unlabeled_loader,
    }

    return dataloaders


def get_dataloaders_unlabeled_psuedo(
    root,
    mapping_file=UNLABELED_JSON_FILE,
    map_keys="all",
    batch_size=8,
    shuffle=True,
    num_workers=5,
    use_cache=False,
):

    # Get list of data dictionaries from decoded paths
    unlabeled_psuedo_dicts = path_decoder(
        root, mapping_file, map_keys, no_label=False, unlabeled=True
    )

    # Use caching for non-random transforms when use_cache
    DATACLASS = CacheDataset if use_cache else Dataset

    # Obtain datasets with transforms
    unlabeled_psuedo_dicts, _ = split_train_valid(
        unlabeled_psuedo_dicts, valid_portion=0
    )
    unlabeled_psuedo_set = DATACLASS(unlabeled_psuedo_dicts, transform=train_transforms)

    # Set dataloader for Unlabeled dataset
    unlabeled_psuedo_loader = DataLoader(
        unlabeled_psuedo_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    dataloaders = {"unlabeled": unlabeled_psuedo_loader}

    return dataloaders