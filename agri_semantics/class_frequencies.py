import subprocess
from os.path import abspath, dirname, join

import click
import torch
import yaml
from agri_semantics.datasets import get_data_module


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=join(dirname(abspath(__file__)), "config", "config.yaml"),
)
def main(config):
    with open(config, "r") as config_file:
        cfg = yaml.safe_load(config_file)
    cfg["git_commit_version"] = str(subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip())

    data_module = get_data_module(cfg)
    data_module.setup(stage="fit")

    print(f"------------------------------------- TRAIN DATASET -------------------------------------")
    class_counts = torch.zeros(cfg["model"]["num_classes"])
    for batch in data_module.train_dataloader():
        class_counts += torch.bincount(batch["anno"].flatten(), minlength=cfg["model"]["num_classes"])

    print(f"TOTAL NUMBER OF PIXELS: {class_counts.sum()}")
    print(f"CLASS COUNTS: {class_counts}")
    print(f"CLASS FREQUENCIES: {class_counts / class_counts.sum()}")
    print(f"INVERSE CLASS FREQUENCIES: {class_counts.sum() / class_counts}")

    print(f"------------------------------------- VALIDATION DATASET -------------------------------------")
    class_counts = torch.zeros(cfg["model"]["num_classes"])
    for batch in data_module.val_dataloader():
        class_counts += torch.bincount(batch["anno"].flatten(), minlength=cfg["model"]["num_classes"])

    print(f"TOTAL NUMBER OF PIXELS: {class_counts.sum()}")
    print(f"CLASS COUNTS: {class_counts}")
    print(f"CLASS FREQUENCIES: {class_counts / class_counts.sum()}")
    print(f"INVERSE CLASS FREQUENCIES: {class_counts.sum() / class_counts}")


if __name__ == "__main__":
    main()
