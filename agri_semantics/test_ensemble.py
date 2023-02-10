import click
from os.path import join, dirname, abspath
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
import yaml

from agri_semantics.datasets import get_data_module
from agri_semantics.models import get_model
from agri_semantics.models.models import EnsembleNet


@click.command()
### Add your options here
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=join(dirname(abspath(__file__)), "config", "config.yaml"),
)
@click.option("--checkpoint", "-ckpt", type=str, help="path to checkpoint files base name", required=True)
def main(config: str, checkpoint: str):
    with open(config, "r") as config_file:
        cfg = yaml.safe_load(config_file)

    data = get_data_module(cfg)

    models = [get_model(cfg) for _ in range(cfg["model"]["num_models"])]
    for i in range(cfg["model"]["num_models"]):
        models[i] = models[i].load_from_checkpoint(f"{checkpoint}_model{i}.ckpt", cfg=cfg)

    ensemble_model = EnsembleNet(cfg, models)

    tb_logger = pl_loggers.TensorBoardLogger(
        f"experiments/{cfg['experiment']['id']}", name="ensemble", default_hp_metric=False
    )
    trainer = Trainer(logger=tb_logger, gpus=cfg["train"]["n_gpus"])
    trainer.test(ensemble_model, data)


if __name__ == "__main__":
    main()
