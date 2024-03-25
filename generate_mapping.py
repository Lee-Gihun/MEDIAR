import os, glob
import json
import argparse


def public_paths_labeled(root):
    """Map paths for public datasets as dictionary list"""

    images_raw = sorted(glob.glob(os.path.join(root, "Public/images/*")))
    labels_raw = sorted(glob.glob(os.path.join(root, "Public/labels/*")))

    data_dicts = []

    for image_path, label_path in zip(images_raw, labels_raw):
        name1 = image_path.split("/")[-1].split(".")[0]
        name2 = label_path.split("/")[-1].split("_label")[0]
        assert name1 == name2

        data_item = {
            "img": image_path.split("MEDIAR/")[-1],
            "label": label_path.split("MEDIAR/")[-1],
        }

        data_dicts.append(data_item)

    map_dict = {"public": data_dicts}

    return map_dict


def official_paths_labeled(root):
    """Map paths for official labeled datasets as dictionary list"""

    image_path = os.path.join(root, "Official/Training/images/*")
    label_path = os.path.join(root, "Official/Training/labels/*")

    images_raw = sorted(glob.glob(image_path))
    labels_raw = sorted(glob.glob(label_path))
    data_dicts = []

    for image_path, label_path in zip(images_raw, labels_raw):
        name1 = image_path.split("/")[-1].split(".")[0]
        name2 = label_path.split("/")[-1].split("_label")[0]
        assert name1 == name2

        data_item = {
            "img": image_path.split("MEDIAR/")[-1],
            "label": label_path.split("MEDIAR/")[-1],
        }

        data_dicts.append(data_item)

    map_dict = {"official": data_dicts}

    return map_dict


def official_paths_tuning(root):
    """Map paths for official tuning datasets as dictionary list"""

    image_path = os.path.join(root, "Official/Tuning/images/*")
    images_raw = sorted(glob.glob(image_path))

    data_dicts = []

    for image_path in images_raw:
        data_item = {"img": image_path.split("MEDIAR/")[-1]}
        data_dicts.append(data_item)

    map_dict = {"official": data_dicts}

    return map_dict


def add_mapping_to_json(json_file, map_dict):
    """Save mapped dictionary as a json file"""

    if not os.path.exists(json_file):
        with open(json_file, "w") as file:
            json.dump({}, file)

    with open(json_file, "r") as file:
        data = json.load(file)

    for map_key, map_item in map_dict.items():
        if map_key not in data.keys():
            data[map_key] = map_item
        else:
            print('>>> "{}" already exists in path map keys...'.format(map_key))

    with open(json_file, "w") as file:
        json.dump(data, file)


if __name__ == "__main__":
    # [!Caution] The paths should be overrided for the local environment!
    parser = argparse.ArgumentParser(description="Mapping files and paths")
    parser.add_argument("--root", default=".", type=str)
    args = parser.parse_args()

    MAP_DIR = "./train_tools/data_utils/"

    print("\n----------- Path Mapping for Labeled Data is Started... -----------\n")

    map_labeled = os.path.join(MAP_DIR, "mapping_labeled.json")
    map_dict = official_paths_labeled(args.root)
    add_mapping_to_json(map_labeled, map_dict)

    print("\n----------- Path Mapping for Tuning Data is Started... -----------\n")

    map_labeled = os.path.join(MAP_DIR, "mapping_tuning.json")
    map_dict = official_paths_tuning(args.root)
    add_mapping_to_json(map_labeled, map_dict)

    print("\n----------- Path Mapping for Public Data is Started... -----------\n")

    map_public = os.path.join(MAP_DIR, "mapping_public.json")
    map_dict = public_paths_labeled(args.root)
    add_mapping_to_json(map_public, map_dict)

    print("\n-------------- Path Mapping is Ended !!! ---------------------------\n")
