import torch
import numpy as np
import time, os
import tifffile as tif

from datetime import datetime
from zipfile import ZipFile
from pytz import timezone

from train_tools.data_utils.transforms import get_pred_transforms


class BasePredictor:
    def __init__(
        self,
        model,
        device,
        input_path,
        output_path,
        make_submission=False,
        exp_name=None,
        algo_params=None,
    ):
        self.model = model
        self.device = device
        self.input_path = input_path
        self.output_path = output_path
        self.make_submission = make_submission
        self.exp_name = exp_name

        # Assign algoritm-specific arguments
        if algo_params:
            self.__dict__.update((k, v) for k, v in algo_params.items())

        # Prepare inference environments
        self._setups()

    @torch.no_grad()
    def conduct_prediction(self):
        self.model.to(self.device)
        self.model.eval()
        total_time = 0
        total_times = []

        for img_name in self.img_names:
            img_data = self._get_img_data(img_name)
            img_data = img_data.to(self.device)

            start = time.time()

            pred_mask = self._inference(img_data)
            pred_mask = self._post_process(pred_mask.squeeze(0).cpu().numpy())
            self.write_pred_mask(
                pred_mask, self.output_path, img_name, self.make_submission
            )

            end = time.time()

            time_cost = end - start
            total_times.append(time_cost)
            total_time += time_cost
            print(
                f"Prediction finished: {img_name}; img size = {img_data.shape}; costing: {time_cost:.2f}s"
            )

        print(f"\n Total Time Cost: {total_time:.2f}s")

        if self.make_submission:
            fname = "%s.zip" % self.exp_name

            os.makedirs("./submissions", exist_ok=True)
            submission_path = os.path.join("./submissions", fname)

            with ZipFile(submission_path, "w") as zipObj2:
                pred_names = sorted(os.listdir(self.output_path))
                for pred_name in pred_names:
                    pred_path = os.path.join(self.output_path, pred_name)
                    zipObj2.write(pred_path)

            print("\n>>>>> Submission file is saved at: %s\n" % submission_path)

        return time_cost

    def write_pred_mask(self, pred_mask, output_dir, image_name, submission=False):

        # All images should contain at least 5 cells
        if submission:
            if not (np.max(pred_mask) > 5):
                print("[!Caution] Only %d Cells Detected!!!\n" % np.max(pred_mask))

        file_name = image_name.split(".")[0]
        file_name = file_name + "_label.tiff"
        file_path = os.path.join(output_dir, file_name)

        tif.imwrite(file_path, pred_mask, compression="zlib")

    def _setups(self):
        self.pred_transforms = get_pred_transforms()
        os.makedirs(self.output_path, exist_ok=True)

        now = datetime.now(timezone("Asia/Seoul"))
        dt_string = now.strftime("%m%d_%H%M")
        self.exp_name = (
            self.exp_name + dt_string if self.exp_name is not None else dt_string
        )

        self.img_names = sorted(os.listdir(self.input_path))

    def _get_img_data(self, img_name):
        img_path = os.path.join(self.input_path, img_name)
        img_data = self.pred_transforms(img_path)
        img_data = img_data.unsqueeze(0)

        return img_data

    def _inference(self, img_data):
        raise NotImplementedError

    def _post_process(self, pred_mask):
        raise NotImplementedError
