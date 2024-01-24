import subprocess
from os.path import abspath, dirname, join

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
        raise NotImplementedError(f"No monitoring metric implemented for {task} task!")


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
def main(config, weights, checkpoint):
    with open(config, "r") as config_file:
        cfg = yaml.safe_load(config_file)
    cfg["git_commit_version"] = str(subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip())

    # Load data and model
    data = get_data_module(
        cfg, human_data=cfg["data"]["batch_size"]["human"] > 0, pseudo_data=cfg["data"]["batch_size"]["pseudo"] > 0
    )
    data.setup()
    model = get_model(cfg, num_train_data=len(data.train_dataloader()["human"].dataset))
    print(model)

    if weights:
        model = model.load_from_checkpoint(weights, cfg=cfg)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver = ModelCheckpoint(
        monitor=f"Validation/{monitoring_metric(cfg['model']['task'])}",
        filename=cfg["experiment"]["id"] + "_{epoch:02d}_{iou:.2f}",
        mode=monitoring_mode(cfg["model"]["task"]),
        save_last=True,
    )

    tb_logger = pl_loggers.TensorBoardLogger(f"experiments/{cfg['experiment']['id']}", default_hp_metric=False)

    # Setup trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=cfg["train"]["n_gpus"],
        logger=tb_logger,
        resume_from_checkpoint=checkpoint,
        max_epochs=cfg["train"]["max_epoch"],
        callbacks=[lr_monitor, checkpoint_saver],
    )

    # Train!
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
