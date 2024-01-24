from os.path import join, dirname, abspath

import click
import yaml
from agri_semantics.datasets import get_data_module
from agri_semantics.models import get_model
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers


@click.command()
### Add your options here
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=join(dirname(abspath(__file__)), "config", "config.yaml"),
)
@click.option("--checkpoint", "-ckpt", type=str, help="path to checkpoint file (.ckpt)", required=True)
def main(config, checkpoint):
    with open(config, "r") as config_file:
        cfg = yaml.safe_load(config_file)

    # Load data and model
    data = get_data_module(cfg)

    model = get_model(cfg)
    model = model.load_from_checkpoint(checkpoint, cfg=cfg)

    tb_logger = pl_loggers.TensorBoardLogger("experiments/" + cfg["experiment"]["id"], default_hp_metric=False)

    # Setup trainer
    trainer = Trainer(logger=tb_logger, accelerator="gpu", devices=cfg["train"]["n_gpus"])

    # Test!
    trainer.test(model, data)


if __name__ == "__main__":
    main()
