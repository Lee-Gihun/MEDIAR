import torch
import os
import wandb
import argparse, pprint

from train_tools import *
from SetupDict import TRAINER, OPTIMIZER, SCHEDULER, MODELS, PREDICTOR

# Ignore warnings for tiffle image reading
import logging

logging.getLogger().setLevel(logging.ERROR)

# Set torch base print precision
torch.set_printoptions(6)


def _get_setups(args):
    """Get experiment configuration"""

    # Set model
    model_args = args.train_setups.model
    model = MODELS[model_args.name](**model_args.params)

    # Load pretrained weights
    if model_args.pretrained.enabled:
        weights = torch.load(model_args.pretrained.weights, map_location="cpu")

        print("\nLoading pretrained model....")
        model.load_state_dict(weights, strict=model_args.pretrained.strict)

    # Set dataloaders
    dataloaders = datasetter.get_dataloaders_labeled(**args.data_setups.labeled)

    # Set optimizer
    optimizer_args = args.train_setups.optimizer
    optimizer = OPTIMIZER[optimizer_args.name](
        model.parameters(), **optimizer_args.params
    )

    # Set scheduler
    scheduler = None

    if args.train_setups.scheduler.enabled:
        scheduler_args = args.train_setups.scheduler
        scheduler = SCHEDULER[scheduler_args.name](optimizer, **scheduler_args.params)

    # Set trainer
    trainer_args = args.train_setups.trainer
    trainer = TRAINER[trainer_args.name](
        model, dataloaders, optimizer, scheduler, **trainer_args.params
    )

    # Check if no validation
    if args.data_setups.labeled.valid_portion == 0:
        trainer.no_valid = True

    # Set public dataloader
    if args.data_setups.public.enabled:
        dataloaders = datasetter.get_dataloaders_public(
            **args.data_setups.public.params
        )
        trainer.public_loader = dataloaders["public"]
        trainer.public_iterator = iter(dataloaders["public"])

    return trainer


def main(args):
    """Execute experiment."""

    # Initialize W&B
    wandb.init(config=args, **args.wandb_setups)

    # How many batches to wait before logging training status
    wandb.config.log_interval = 10

    # Fix randomness for reproducibility
    random_seeder(args.train_setups.seed)

    # Set experiment
    trainer = _get_setups(args)

    # Watch parameters & gradients of model
    wandb.watch(trainer.model, log="all", log_graph=True)

    # Conduct experiment
    trainer.train()

    # Upload model to wandb server
    model_path = os.path.join(wandb.run.dir, "model.pth")
    torch.save(trainer.model.state_dict(), model_path)
    wandb.save(model_path)

    # Conduct prediction using the trained model
    predictor = PREDICTOR[args.train_setups.trainer.name](
        trainer.model,
        args.train_setups.trainer.params.device,
        args.pred_setups.input_path,
        args.pred_setups.output_path,
        args.pred_setups.make_submission,
        args.pred_setups.exp_name,
        args.pred_setups.algo_params,
    )

    total_time = predictor.conduct_prediction()
    wandb.log({"total_time": total_time})


# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Config file processing")
parser.add_argument("--config_path", default="./config/baseline.json", type=str)
args = parser.parse_args()

#######################################################################################

if __name__ == "__main__":
    # Load configuration from .json file
    opt = ConfLoader(args.config_path).opt

    # Print configuration dictionary pretty
    pprint_config(opt)

    # Run experiment
    main(opt)
