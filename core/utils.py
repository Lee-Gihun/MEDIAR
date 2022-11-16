import torch
import wandb
import pprint

__all__ = ["print_learning_device", "print_with_logging"]


def print_learning_device(device):
    """Get and print the learning device information."""
    if device == "cpu":
        device_name = device

    else:
        if isinstance(device, str):
            device_idx = int(device[-1])
        elif isinstance(device, torch._device):
            device_idx = device.index

        device_name = torch.cuda.get_device_name(device_idx)

    print("")
    print("=" * 50)
    print("Train start on device: {}".format(device_name))
    print("=" * 50)


def print_with_logging(results, step):
    """Print and log on the W&B server.

    Args:
        results (dict): results dictionary
        step (int): epoch index
    """
    # Print the results dictionary
    pp = pprint.PrettyPrinter(compact=True)
    pp.pprint(results)
    print()

    # Log on the w&b server
    wandb.log(results, step=step)
