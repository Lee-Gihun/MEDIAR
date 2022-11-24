import os
import json
import numpy as np

__all__ = ["split_train_valid", "path_decoder"]


def split_train_valid(data_dicts, valid_portion=0.1):
    """Split train/validata data according to the given proportion"""

    train_dicts, valid_dicts = data_dicts, []
    if valid_portion > 0:

        # Obtain & shuffle data indices
        num_data_dicts = len(data_dicts)
        indices = np.arange(num_data_dicts)
        np.random.shuffle(indices)

        # Divide train/valid indices by the proportion
        valid_size = int(num_data_dicts * valid_portion)
        train_indices = indices[valid_size:]
        valid_indices = indices[:valid_size]

        # Assign data dicts by split indices
        train_dicts = [data_dicts[idx] for idx in train_indices]
        valid_dicts = [data_dicts[idx] for idx in valid_indices]

    print(
        "\n(DataLoaded) Training data size: %d, Validation data size: %d\n"
        % (len(train_dicts), len(valid_dicts))
    )

    return train_dicts, valid_dicts


def path_decoder(root, mapping_file, no_label=False, unlabeled=False):
    """Decode img/label file paths from root & mapping directory.

    Args:
        root (str):
        mapping_file (str): json file containing image & label file paths.
        no_label (bool, optional): whether to include "label" key. Defaults to False.

    Returns:
        list: list of dictionary. (ex. [{"img": img_path, "label": label_path}, ...])
    """

    data_dicts = []

    with open(mapping_file, "r") as file:
        data = json.load(file)

        for map_key in data.keys():

            # If no_label, assign "img" key only
            if no_label:
                data_dict_item = [
                    {"img": os.path.join(root, elem["img"]),} for elem in data[map_key]
                ]

            # If label exists, assign both "img" and "label" keys
            else:
                data_dict_item = [
                    {
                        "img": os.path.join(root, elem["img"]),
                        "label": os.path.join(root, elem["label"]),
                    }
                    for elem in data[map_key]
                ]

            # Add refined datasets to be returned
            data_dicts += data_dict_item

    if unlabeled:
        refined_data_dicts = []

        # Exclude the corrupted image to prevent errror
        for data_dict in data_dicts:
            if "00504" not in data_dict["img"]:
                refined_data_dicts.append(data_dict)

        data_dicts = refined_data_dicts

    return data_dicts
