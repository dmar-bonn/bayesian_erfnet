import os
import subprocess
from typing import Dict

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from agri_semantics.active_learning import get_active_learner
from agri_semantics.constants import ActiveLearners


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml"),
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
def main(config: str, weights: str, checkpoint: str):
    with open(config, "r") as config_file:
        cfg = yaml.safe_load(config_file)

    cfg["git_commit_version"] = str(subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip())

    if cfg["active_learning"]["type"] == ActiveLearners.ALL:
        active_learner_results = {}
        for active_learner_type in [ActiveLearners.RANDOM, ActiveLearners.BALD]:
            active_learner = get_active_learner(cfg, weights, checkpoint, active_learner_type)
            active_learner.run()
            active_learner_results[active_learner_type] = active_learner.test_statistics
        plot_experiment_results(cfg, active_learner_results)
    else:
        active_learner = get_active_learner(cfg, weights, checkpoint)
        active_learner.run()


def plot_experiment_results(cfg: Dict, active_learner_results: Dict):
    experiment_results_df = pd.DataFrame(
        {"method": [], "num_data_points": [], "test_loss": [], "test_iou": [], "test_acc": [], "test_f1": []}
    )
    tmp_results_df = pd.DataFrame(
        {"method": [], "num_data_points": [], "test_loss": [], "test_iou": [], "test_acc": [], "test_f1": []}
    )
    for active_learner_type in active_learner_results.keys():
        test_losses = []
        test_ious = []
        test_f1_scores = []
        test_accs = []
        num_data_points_list = list(active_learner_results[active_learner_type].keys())
        for num_data_points in active_learner_results[active_learner_type].keys():
            test_losses.append(active_learner_results[active_learner_type][num_data_points]["Test/Loss"])
            test_ious.append(active_learner_results[active_learner_type][num_data_points]["Test/mIoU"])
            test_f1_scores.append(active_learner_results[active_learner_type][num_data_points]["Test/F1"])
            test_accs.append(active_learner_results[active_learner_type][num_data_points]["Test/Acc"])

        tmp_results_df["num_data_points"] = num_data_points_list
        tmp_results_df["test_loss"] = test_losses
        tmp_results_df["test_iou"] = test_ious
        tmp_results_df["test_f1"] = test_f1_scores
        tmp_results_df["test_acc"] = test_accs
        tmp_results_df["method"] = active_learner_type
        experiment_results_df = experiment_results_df.append(tmp_results_df, ignore_index=True)

    for performance_metric in ["test_loss", "test_f1", "test_acc", "test_iou"]:
        ax = sns.lineplot(
            x="num_data_points",
            y=performance_metric,
            hue="method",
            data=experiment_results_df,
            linewidth=3,
        )
        ax.set_title(f"{performance_metric} Performance")
        ax.set_ylabel(f"{performance_metric}")
        ax.set_xlabel("Number of Train Data Points")

        plt.tight_layout()
        plt.savefig(
            os.path.join("experiments", cfg["experiment"]["id"], f"{performance_metric}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.clf()


if __name__ == "__main__":
    main()
