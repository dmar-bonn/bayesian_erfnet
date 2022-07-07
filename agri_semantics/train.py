import subprocess
from os.path import abspath, dirname, join

import click
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from agri_semantics.datasets import get_data_module
from agri_semantics.models import get_model

##############################################################################################
#                                                                                            #
#  Pytorch Lightning ERFNet training wrapper from Jan Weyler. Our Bayesian-ERFNet            #
#  training builds upon Jan's ERFNet implementation.                                         #
#                                                                                            #
##############################################################################################


@click.command()
### Add your options here
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=join(dirname(abspath(__file__)), "config", "config.yaml"),
)
@click.option(
    "--weights",
    "-w",
    type=str,
    help="path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.",
    default=None,
)
@click.option(
    "--checkpoint", "-ckpt", type=str, help="path to checkpoint file (.ckpt) to resume training.", default=None
)
def main(config, weights, checkpoint):
    with open(config, "r") as config_file:
        cfg = yaml.safe_load(config_file)
    cfg["git_commit_version"] = str(subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip())

    # Load data and model
    data = get_data_module(cfg)
    model = get_model(cfg)
    print(model)

    if weights:
        model = model.load_from_checkpoint(weights, hparams=cfg)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver = ModelCheckpoint(
        monitor="val:iou", filename=cfg["experiment"]["id"] + "_{epoch:02d}_{iou:.2f}", mode="max", save_last=True
    )

    tb_logger = pl_loggers.TensorBoardLogger(f"experiments/{cfg['experiment']['id']}", default_hp_metric=False)

    # Setup trainer
    trainer = Trainer(
        gpus=cfg["train"]["n_gpus"],
        logger=tb_logger,
        resume_from_checkpoint=checkpoint,
        max_epochs=cfg["train"]["max_epoch"],
        callbacks=[lr_monitor, checkpoint_saver],
    )

    # Train!
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
