from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def iou_from_conf_matrices(conf_matrices: List[torch.Tensor], ignore_index: int = None) -> float:
    def conf_mat_masked(conf_mat):
        if ignore_index is None:
            return conf_mat

        conf_mat[ignore_index] = 0.0
        return conf_mat

    conf_matrices = [conf_mat_masked(conf_mat) for conf_mat in conf_matrices]

    intersections = [torch.diag(conf_mat) for conf_mat in conf_matrices]
    unions = [conf_mat.sum(0) + conf_mat.sum(1) - intersections[i] for i, conf_mat in enumerate(conf_matrices)]

    total_intersection = torch.sum(torch.stack(intersections), dim=0)
    total_union = torch.sum(torch.stack(unions), dim=0)
    iou = total_intersection / total_union
    iou[total_union == 0] = 0

    if ignore_index is not None:
        iou = torch.cat([iou[:ignore_index], iou[ignore_index + 1 :]])

    return torch.mean(iou).item()


def accuracy_from_conf_matrices(conf_matrices: List[torch.Tensor], ignore_index: int = None) -> float:
    def conf_mat_masked(conf_mat):
        if ignore_index is None:
            return conf_mat

        conf_mat[ignore_index, :] = 0.0
        conf_mat[:, ignore_index] = 0.0
        return conf_mat

    true_predictions = torch.stack([torch.sum(torch.diag(conf_mat_masked(conf_mat))) for conf_mat in conf_matrices])
    num_predictions = torch.stack([torch.sum(conf_mat_masked(conf_mat)) for conf_mat in conf_matrices])

    return torch.sum(true_predictions) / torch.sum(num_predictions)


def total_conf_matrix_from_conf_matrices(conf_matrices: List[torch.Tensor]) -> torch.Tensor:
    total_conf_matrix = torch.sum(torch.stack(conf_matrices), dim=0) / len(conf_matrices)
    total_conf_matrix = total_conf_matrix / torch.sum(total_conf_matrix, dim=1, keepdim=True)
    return total_conf_matrix


def compute_calibration_info(
    prediction_probs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 20,
) -> Dict:
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, dtype=torch.float)
    confidences, predictions = torch.transpose(prediction_probs, 1, -1).flatten(0, -2).max(dim=1)
    accuracies = predictions.eq(targets.flatten())
    confidences, accuracies = confidences.float(), accuracies.float()

    conf_bin = torch.zeros_like(bin_boundaries)
    acc_bin = torch.zeros_like(bin_boundaries)
    prop_bin = torch.zeros_like(bin_boundaries)
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().sum()
        if prop_in_bin.item() > 0:
            acc_bin[i] = accuracies[in_bin].float().sum()
            conf_bin[i] = confidences[in_bin].sum()
            prop_bin[i] = prop_in_bin

    return {"conf_bin": conf_bin, "acc_bin": acc_bin, "prop_bin": prop_bin}


def ece_from_calibration_info(calibration_info_list: List, num_bins: int = 20) -> float:
    conf_bin = torch.zeros(num_bins + 1)
    acc_bin = torch.zeros(num_bins + 1)
    prop_bin = torch.zeros(num_bins + 1)

    for calibration_info in calibration_info_list:
        conf_bin += calibration_info["conf_bin"]
        acc_bin += calibration_info["acc_bin"]
        prop_bin += calibration_info["prop_bin"]

    acc_bin = acc_bin / prop_bin
    conf_bin = conf_bin / prop_bin
    acc_bin[prop_bin == 0] = 0.0
    conf_bin[prop_bin == 0] = 0.0

    return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin / torch.sum(prop_bin)).item()


def compute_calibration_plots(outputs: List[Dict], num_bins: int = 20):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, dtype=torch.float)
    calibration_info_list = [tmp["calibration_info"] for tmp in outputs]
    conf_bin = torch.zeros(num_bins + 1)
    acc_bin = torch.zeros(num_bins + 1)
    prop_bin = torch.zeros(num_bins + 1)

    for calibration_info in calibration_info_list:
        conf_bin += calibration_info["conf_bin"]
        acc_bin += calibration_info["acc_bin"]
        prop_bin += calibration_info["prop_bin"]

    conf_bin = conf_bin / prop_bin
    acc_bin = acc_bin / prop_bin
    acc_bin[prop_bin == 0] = 0.0
    conf_bin[prop_bin == 0] = 0.0

    ax = sns.lineplot(
        x=bin_boundaries.tolist(), y=bin_boundaries.tolist(), color="gray", linestyle="dashed", linewidth=3
    )
    sns.lineplot(x=conf_bin.tolist(), y=acc_bin.tolist(), color="red", linewidth=3, ax=ax)
    ax.set_ylabel("Accuracy", fontsize=15)
    ax.set_xlabel("Confidence", fontsize=15)
    ax.tick_params(labelsize=15)
    fig_ = ax.get_figure()
    plt.close(fig_)

    return fig_
