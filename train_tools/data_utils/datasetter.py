from torch.utils.data import DataLoader
from monai.data import Dataset
import pickle

from .transforms import (
    train_transforms,
    public_transforms,
    valid_transforms,
    tuning_transforms,
    unlabeled_transforms,
)
from .utils import split_train_valid, path_decoder

DATA_LABEL_DICT_PICKLE_FILE = "./train_tools/data_utils/custom/modalities.pkl"

__all__ = [
    "get_dataloaders_labeled",
    "get_dataloaders_public",
    "get_dataloaders_unlabeled",
]


def get_dataloaders_labeled(
    root,
    mapping_file,
    mapping_file_tuning,
    join_mapping_file=None,
    valid_portion=0.0,
    batch_size=8,
    amplified=False,
    relabel=False,
):
    """Set DataLoaders for labeled datasets.

    Args:
        root (str): root directory
        mapping_file (str): json file for mapping dataset
        valid_portion (float, optional): portion of valid datasets. Defaults to 0.1.
        batch_size (int, optional): batch size. Defaults to 8.
        shuffle (bool, optional): shuffles dataloader. Defaults to True.
        num_workers (int, optional): number of workers for each datalaoder. Defaults to 5.

    Returns:
        dict: dictionary of data loaders.
    """

    # Get list of data dictionaries from decoded paths
    data_dicts = path_decoder(root, mapping_file)
    tuning_dicts = path_decoder(root, mapping_file_tuning, no_label=True)

    if amplified:
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
                for i in range(50):
                    data_dicts.append(data_points[i % len_data_points])

    data_transforms = train_transforms

    if join_mapping_file is not None:
        data_dicts += path_decoder(root, join_mapping_file)
        data_transforms = public_transforms

    if relabel:
        for elem in data_dicts:
            cell_idx = int(elem["label"].split("_label.tiff")[0].split("_")[-1])
            if cell_idx in range(340, 499):
                new_label = elem["label"].replace(
                    "/data/CellSeg/Official/Train_Labeled/labels/",
                    "/CellSeg/pretrained_train_ext/",
                )
                elem["label"] = new_label

    # Split datasets as Train/Valid
    train_dicts, valid_dicts = split_train_valid(
        data_dicts, valid_portion=valid_portion
    )

    # Obtain datasets with transforms
    trainset = Dataset(train_dicts, transform=data_transforms)
    validset = Dataset(valid_dicts, transform=valid_transforms)
    tuningset = Dataset(tuning_dicts, transform=tuning_transforms)

    # Set dataloader for Trainset
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=5
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
    root, mapping_file, valid_portion=0.0, batch_size=8,
):
    """Set DataLoaders for labeled datasets.

    Args:
        root (str): root directory
        mapping_file (str): json file for mapping dataset
        valid_portion (float, optional): portion of valid datasets. Defaults to 0.1.
        batch_size (int, optional): batch size. Defaults to 8.
        shuffle (bool, optional): shuffles dataloader. Defaults to True.

    Returns:
        dict: dictionary of data loaders.
    """

    # Get list of data dictionaries from decoded paths
    data_dicts = path_decoder(root, mapping_file)

    # Split datasets as Train/Valid
    train_dicts, _ = split_train_valid(data_dicts, valid_portion=valid_portion)

    trainset = Dataset(train_dicts, transform=public_transforms)
    # Set dataloader for Trainset
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=5
    )

    # Form dataloaders as dictionary
    dataloaders = {
        "public": train_loader,
    }

    return dataloaders


def get_dataloaders_unlabeled(
    root, mapping_file, batch_size=8, shuffle=True, num_workers=5,
):
    """Set dataloaders for unlabeled dataset."""
    # Get list of data dictionaries from decoded paths
    unlabeled_dicts = path_decoder(root, mapping_file, no_label=True, unlabeled=True)

    # Obtain datasets with transforms
    unlabeled_dicts, _ = split_train_valid(unlabeled_dicts, valid_portion=0)
    unlabeled_set = Dataset(unlabeled_dicts, transform=unlabeled_transforms)

    # Set dataloader for Unlabeled dataset
    unlabeled_loader = DataLoader(
        unlabeled_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    dataloaders = {
        "unlabeled": unlabeled_loader,
    }

    return dataloaders


def get_dataloaders_unlabeled_psuedo(
    root, mapping_file, batch_size=8, shuffle=True, num_workers=5,
):

    # Get list of data dictionaries from decoded paths
    unlabeled_psuedo_dicts = path_decoder(
        root, mapping_file, no_label=False, unlabeled=True
    )

    # Obtain datasets with transforms
    unlabeled_psuedo_dicts, _ = split_train_valid(
        unlabeled_psuedo_dicts, valid_portion=0
    )
    unlabeled_psuedo_set = Dataset(unlabeled_psuedo_dicts, transform=train_transforms)

    # Set dataloader for Unlabeled dataset
    unlabeled_psuedo_loader = DataLoader(
        unlabeled_psuedo_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    dataloaders = {"unlabeled": unlabeled_psuedo_loader}

    return dataloaders
