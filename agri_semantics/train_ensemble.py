import subprocess
from os.path import abspath, dirname, join
from typing import Dict

import click
import yaml
from agri_semantics.datasets import get_data_module
from agri_semantics.models import get_model
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


##############################################################################################
#                                                                                            #
#  Pytorch Lightning ERFNet training wrapper from Jan Weyler. Our Bayesian-ERFNet            #
#  training builds upon Jan's ERFNet implementation.                                         #
#                                                                                            #
##############################################################################################


def monitoring_metric(task) -> str:
    if task == "classification":
        return "mIoU"
    elif task == "regression":
        return "RMSE"
    else:
        raise NotImplementedError(f"No early stopping metric implemented for {task} task!")


def monitoring_mode(task) -> str:
    if task == "classification":
        return "max"
    elif task == "regression":
        return "min"
    else:
        raise NotImplementedError(f"No monitoring mode implemented for {task} task!")


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
def main(config: str, weights: str = None, checkpoint: str = None):
    with open(config, "r") as config_file:
        cfg = yaml.safe_load(config_file)
    cfg["git_commit_version"] = str(subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip())

    data_modules = [get_data_module(cfg) for _ in range(cfg["model"]["num_models"])]
    models = [get_model(cfg) for _ in range(cfg["model"]["num_models"])]

    if weights:
        for i in range(cfg["model"]["num_models"]):
            models[i] = models[i].load_from_checkpoint(f"{weights}_model{i}.ckpt", cfg=cfg)

    trainers = [setup_trainer(i, cfg, checkpoint) for i in range(cfg["model"]["num_models"])]
    for i in range(cfg["model"]["num_models"]):
        print("-------------------------------------------------------------------")
        print(f"TRAIN {cfg['model']['name']} model {i}")
        trainers[i].fit(models[i], data_modules[i])


def setup_trainer(model_id: int, cfg: Dict, checkpoint: str) -> Trainer:
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver = ModelCheckpoint(
        monitor=f"Validation/{monitoring_metric(cfg['model']['task'])}",
        filename=f"{cfg['experiment']['id']}_model{model_id}_best",
        mode=monitoring_mode(cfg["model"]["task"]),
        save_last=True,
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        f"experiments/{cfg['experiment']['id']}", name=f"{cfg['model']['name']}_{model_id}", default_hp_metric=False
    )

    tmp_checkpoint = f"{checkpoint}_model{model_id}.ckpt" if checkpoint is not None else None
    trainer = Trainer(
        accelerator="gpu",
        devices=cfg["train"]["n_gpus"],
        logger=tb_logger,
        resume_from_checkpoint=tmp_checkpoint,
        max_epochs=cfg["train"]["max_epoch"],
        callbacks=[lr_monitor, checkpoint_saver],
        log_every_n_steps=1,
    )

    return trainer


if __name__ == "__main__":
    main()
