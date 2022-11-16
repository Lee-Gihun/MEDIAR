import torch
import argparse, pprint

from train_tools import *
from SetupDict import MODELS, PREDICTOR

# Set torch base print precision
torch.set_printoptions(6)


def main(args):
    """Execute experiment."""

    model_args = args.pred_setups.model
    model = MODELS[model_args.name](**model_args.params)

    model.load_state_dict(torch.load(args.pred_setups.model_path, map_location="cpu"))

    predictor = PREDICTOR[args.pred_setups.name](
        model,
        args.pred_setups.device,
        args.pred_setups.input_path,
        args.pred_setups.output_path,
        args.pred_setups.make_submission,
    )

    _ = predictor.conduct_prediction()


# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Config file processing")
parser.add_argument("--config_path", default="./config/pred/baseline.json", type=str)
args = parser.parse_args()

#######################################################################################

if __name__ == "__main__":
    # Load configuration from .json file
    opt = ConfLoader(args.config_path).opt

    # Print configuration dictionary pretty
    print("")
    print("=" * 50 + " Configuration " + "=" * 50)
    pp = pprint.PrettyPrinter(compact=True)
    pp.pprint(opt)
    print("=" * 120)

    # Run experiment
    main(opt)
