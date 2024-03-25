import numpy as np
import pandas as pd
import tifffile as tif
import argparse
import os
from collections import OrderedDict
from tqdm import tqdm

from train_tools.measures import evaluate_f1_score_cellseg


def main():
    ### Directory path arguments ###
    parser = argparse.ArgumentParser("Compute F1 score for cell segmentation results")
    parser.add_argument(
        "--gt_path",
        type=str,
        help="path to ground truth; file names end with _label.tiff",
        required=True,
    )
    parser.add_argument(
        "--pred_path", type=str, help="path to segmentation results", required=True
    )
    parser.add_argument("--save_path", default=None, help="path where to save metrics")

    args = parser.parse_args()

    # Get files from the paths
    gt_path, pred_path = args.gt_path, args.pred_path
    names = sorted(os.listdir(pred_path))

    names_total = []
    precisions_total, recalls_total, f1_scores_total = [], [], []

    for name in tqdm(names):
        assert name.endswith(
            "_label.tiff"
        ), "The suffix of label name should be _label.tiff"

        # Load the images
        gt = tif.imread(os.path.join(gt_path, name))
        pred = tif.imread(os.path.join(pred_path, name))

        # Evaluate metrics
        precision, recall, f1_score = evaluate_f1_score_cellseg(gt, pred, threshold=0.5)

        names_total.append(name)
        precisions_total.append(np.round(precision, 4))
        recalls_total.append(np.round(recall, 4))
        f1_scores_total.append(np.round(f1_score, 4))

    # Refine data as dataframe
    cellseg_metric = OrderedDict()
    cellseg_metric["Names"] = names_total
    cellseg_metric["Precision"] = precisions_total
    cellseg_metric["Recall"] = recalls_total
    cellseg_metric["F1_Score"] = f1_scores_total

    cellseg_metric = pd.DataFrame(cellseg_metric)
    print("mean F1 Score:", np.mean(cellseg_metric["F1_Score"]))

    # Save results
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
        cellseg_metric.to_csv(
            os.path.join(args.save_path, "seg_metric.csv"), index=False
        )


if __name__ == "__main__":
    main()
