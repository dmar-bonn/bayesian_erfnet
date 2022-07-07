import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from pytorch_lightning.core.lightning import LightningModule

import agri_semantics.models.blocks as blocks
import agri_semantics.utils.utils as utils
from agri_semantics.constants import Losses
from agri_semantics.models.loss import CrossEntropyLoss
from agri_semantics.utils import metrics


##############################################################################################
#                                                                                            #
#  Pytorch Lightning ERFNet implementation from Jan Weyler. Our Bayesian-ERFNet              #
#  implementation builds upon Jan's ERFNet implementation.                                   #
#                                                                                            #
##############################################################################################

IGNORE_INDEX = {"cityscapes": 19, "weedmap": -100, "rit18": 0, "rit18_merged": 0, "potsdam": 0}


def get_criterion(cfg, weight: torch.Tensor = None):
    loss_name = cfg["model"]["loss"]
    if loss_name == Losses.CROSS_ENTROPY:
        return CrossEntropyLoss(ignore_index=IGNORE_INDEX[cfg["data"]["name"]], weight=weight)
    else:
        raise RuntimeError("Loss {} not available".format(loss_name))


class BayesianERFNet(LightningModule):
    def __init__(self, hparams: Dict, num_train_data: int = 1):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = self.hparams["hparams"]["model"]["num_classes"]
        num_classes_pretrained = self.hparams["hparams"]["model"]["num_classes"]
        if "num_classes_pretrained" in self.hparams["hparams"]["model"]:
            num_classes_pretrained = self.hparams["hparams"]["model"]["num_classes_pretrained"]
        num_classes_init = num_classes_pretrained

        epistemic_version = "standard"
        if "epistemic_version" in self.hparams["hparams"]["model"]:
            epistemic_version = self.hparams["hparams"]["model"]["epistemic_version"]

        in_channels = self.hparams["hparams"]["model"]["in_channels"]
        dropout_prob = self.hparams["hparams"]["model"]["dropout_prob"]
        shared_decoder = self.hparams["hparams"]["model"]["shared_decoder"]
        deep_encoder = self.hparams["hparams"]["model"]["deep_encoder"]

        self.num_mc_aleatoric = self.hparams["hparams"]["train"]["num_mc_aleatoric"]
        self.num_mc_epistemic = self.hparams["hparams"]["train"]["num_mc_epistemic"]

        self.model = BayesianERFNetModel(
            num_classes_init, in_channels, dropout_prob, shared_decoder, deep_encoder, epistemic_version
        )
        self.active_learning = "active_learning" in self.hparams["hparams"]
        self.num_train_data = num_train_data

        self.inv_class_frequencies = None
        if "class_frequencies" in self.hparams["hparams"]["model"]:
            class_frequencies = torch.Tensor(self.hparams["hparams"]["model"]["class_frequencies"])
            self.inv_class_frequencies = class_frequencies.sum() / class_frequencies

    def replace_output_layer(self):
        if self.model.use_shared_decoder:
            self.model.shared_decoder.output_conv = nn.ConvTranspose2d(
                16, self.num_classes + 1, 2, stride=2, padding=0, output_padding=0, bias=True
            )
        else:
            self.model.segmentation_decoder.output_conv = nn.ConvTranspose2d(
                16, self.num_classes, 2, stride=2, padding=0, output_padding=0, bias=True
            )

    def get_loss(self, seg: torch.Tensor, std: torch.Tensor, true_seg: torch.Tensor, device: torch.device = None):
        device = self.device if device is None else device
        if self.inv_class_frequencies is not None:
            self.inv_class_frequencies = self.inv_class_frequencies.to(device)

        loss_fn = get_criterion(self.hparams["hparams"], self.inv_class_frequencies)
        sampled_predictions = torch.zeros((self.num_mc_aleatoric, *seg.size()), device=device)

        for i in range(self.num_mc_aleatoric):
            noise_mean = torch.zeros(seg.size(), device=device)
            noise_std = torch.ones(seg.size(), device=device)
            epsilon = torch.distributions.normal.Normal(noise_mean, noise_std).sample()
            sampled_seg = seg + torch.mul(std, epsilon)
            sampled_predictions[i] = sampled_seg

        mean_prediction = torch.mean(sampled_predictions, dim=0)
        return loss_fn(mean_prediction, true_seg)

    def forward(self, x: torch.Tensor):
        output_seg, output_std = self.model(x)
        return output_seg, output_std

    def compute_aleatoric_uncertainty(self, seg: torch.Tensor, std: torch.Tensor):
        predictions = []
        softmax = nn.Softmax(dim=1)
        for i in range(self.num_mc_aleatoric):
            noise_mean = torch.zeros(seg.size(), device=self.device)
            noise_std = torch.ones(seg.size(), device=self.device)
            epsilon = torch.distributions.normal.Normal(noise_mean, noise_std).sample()
            sampled_seg = seg + torch.mul(std, epsilon)
            predictions.append(softmax(sampled_seg).cpu().numpy())

        mean_predictions = np.mean(predictions, axis=0)
        return -np.sum(mean_predictions * np.log(mean_predictions + sys.float_info.min), axis=1)

    def track_predictions(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        stage: str = "Train",
        dist: Tuple[torch.Tensor, torch.Tensor] = None,
    ):
        sample_img_out = predictions[:1]
        sample_img_out = utils.toOneHot(sample_img_out, self.hparams["hparams"]["data"]["name"])
        self.logger.experiment.add_image(
            f"{stage}/Output image", torch.from_numpy(sample_img_out), 0, dataformats="HWC"
        )

        sample_img_in = images[:1]
        grid = torchvision.utils.make_grid(sample_img_in)
        self.logger.experiment.add_image(f"{stage}/Input image", grid, 0)

        sample_anno = targets[:1]
        sample_anno = utils.toOneHot(sample_anno, self.hparams["hparams"]["data"]["name"])
        self.logger.experiment.add_image(f"{stage}/Annotation", torch.from_numpy(sample_anno), 0, dataformats="HWC")

        if dist is not None:
            sample_aleatoric_unc_out = self.compute_aleatoric_uncertainty(dist[0], dist[1])[0, :, :]
            fig = plt.figure()
            plt.axis("off")
            plt.imshow(sample_aleatoric_unc_out, cmap="plasma")
            self.logger.experiment.add_figure(f"{stage}/Uncertainty/Aleatoric", fig, 0)

    def track_epistemic_uncertainty_stats(self, outputs: List[torch.Tensor]):
        per_class_ep_uncertainties = torch.stack([tmp["per_class_ep_uncertainty"] for tmp in outputs])
        per_class_ep_uncertainty = torch.mean(per_class_ep_uncertainties, dim=0)

        plt.figure(figsize=(10, 7))
        fig_ = sns.barplot(x=list(range(self.num_classes)), y=per_class_ep_uncertainty.tolist()).get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("UncertaintyStats/EpistemicPerClass", fig_, self.current_epoch)
        print(f"Mean per-class epistemic uncertainty: {per_class_ep_uncertainty}")

    def track_confusion_matrix(self, conf_matrices: List[torch.Tensor], stage: str = "Validation"):
        print("Confusion matrix:")
        total_conf_matrix = metrics.total_conf_matrix_from_conf_matrices(conf_matrices)
        df_cm = pd.DataFrame(
            total_conf_matrix.cpu().numpy(), index=range(self.num_classes), columns=range(self.num_classes)
        )
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure(f"ConfusionMatrix/{stage}", fig_, self.current_epoch)

    def track_gradient_norms(self):
        total_grad_norm = 0
        for params in self.model.parameters():
            if params.grad is not None:
                total_grad_norm += params.grad.data.norm(2).item()

        self.log(f"LossStats/GradientNorm", total_grad_norm)

    def track_aleatoric_stats(self, std: torch.Tensor):
        self.log("Variance/TrainMin", torch.min(std))
        self.log("Variance/TrainMax", torch.max(std))
        self.log("Variance/TrainMean", torch.mean(std))

    def training_step(self, batch: Dict, batch_idx):
        est_seg, est_std = self.forward(batch["data"])
        loss = self.get_loss(est_seg, est_std, batch["anno"])
        _, preds = torch.max(est_seg, dim=1)

        # self.track_predictions(batch["data"], preds, batch["anno"], stage="Train")
        self.track_aleatoric_stats(est_std)
        self.track_gradient_norms()
        self.log("train:loss", loss)

        return loss

    def validation_step(self, batch: dict, batch_idx):
        est_seg, est_std = self.forward(batch["data"])
        loss = self.get_loss(est_seg, est_std, batch["anno"])
        _, preds = torch.max(est_seg, dim=1)

        self.log("val:loss", loss, prog_bar=True)
        self.track_aleatoric_stats(est_std)
        # self.track_predictions(batch["data"], preds, batch["anno"], stage="Validation", dist=(est_seg, est_std))

        confusion_matrix = torchmetrics.functional.confusion_matrix(
            preds, batch["anno"], num_classes=self.num_classes, normalize=None
        )

        return {"conf_matrix": confusion_matrix}

    def validation_epoch_end(self, outputs):
        conf_matrices = [tmp["conf_matrix"] for tmp in outputs]
        iou = metrics.iou_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.hparams["hparams"]["data"]["name"]]
        )
        accuracy = metrics.accuracy_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.hparams["hparams"]["data"]["name"]]
        )

        self.log("val:acc", accuracy, prog_bar=True)
        self.log("val:iou", iou, prog_bar=True)
        self.track_confusion_matrix(conf_matrices, stage="Validation")

    def test_step(self, batch: dict, batch_idx):
        est_seg, est_std = self.forward(batch["data"])
        loss = self.get_loss(est_seg, est_std, batch["anno"])

        mean_prediction_probs, _, _, mutual_info_predictions = utils.get_mc_dropout_predictions(
            self,
            batch,
            self.num_mc_epistemic,
            aleatoric_model=True,
            num_mc_aleatoric=self.num_mc_aleatoric,
            device=self.device,
        )
        mean_prediction_probs, mutual_info_predictions = torch.from_numpy(mean_prediction_probs).to(
            self.device
        ), torch.from_numpy(mutual_info_predictions).to(self.device)
        _, preds = torch.max(mean_prediction_probs, dim=1)

        confusion_matrix = torchmetrics.functional.confusion_matrix(
            preds, batch["anno"], num_classes=self.num_classes, normalize=None
        )

        calibration_info = metrics.compute_calibration_info(mean_prediction_probs, batch["anno"], num_bins=20)

        per_class_ep_uncertainty = torch.zeros(self.num_classes)
        for predicted_class in torch.unique(preds):
            mask = preds == predicted_class
            per_class_ep_uncertainty[predicted_class.item()] = torch.mean(mutual_info_predictions[mask]).item()

        self.log("Test/Loss", loss, prog_bar=True)

        return {
            "conf_matrix": confusion_matrix,
            "per_class_ep_uncertainty": per_class_ep_uncertainty,
            "calibration_info": calibration_info,
        }

    def test_epoch_end(self, outputs):
        conf_matrices = [tmp["conf_matrix"] for tmp in outputs]
        calibration_info_list = [tmp["calibration_info"] for tmp in outputs]

        iou = metrics.iou_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.hparams["hparams"]["data"]["name"]]
        )
        accuracy = metrics.accuracy_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.hparams["hparams"]["data"]["name"]]
        )
        ece = metrics.ece_from_calibration_info(calibration_info_list, num_bins=20)

        fig_ = metrics.compute_calibration_plots(outputs)
        self.logger.experiment.add_figure("UncertaintyStats/Calibration", fig_, self.current_epoch)

        self.log("Test/Acc", accuracy, prog_bar=True)
        self.log("Test/IoU", iou, prog_bar=True)
        self.log("Test/ECE", ece, prog_bar=True)

        self.track_epistemic_uncertainty_stats(outputs)
        self.track_confusion_matrix(conf_matrices, stage="Test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["hparams"]["train"]["lr"],
            weight_decay=self.weight_decay,
        )
        return optimizer

    @property
    def weight_decay(self) -> float:
        if not self.active_learning:
            return self.hparams["hparams"]["train"]["weight_decay"]

        return (1 - self.hparams["hparams"]["model"]["dropout_prob"]) / (2 * self.num_train_data)


class ERFNet(LightningModule):
    def __init__(self, hparams: Dict, num_train_data: int = 1):
        super().__init__()
        # name you hyperparameter hparams, then it will be saved automagically.
        self.save_hyperparameters()

        self.num_classes = self.hparams["hparams"]["model"]["num_classes"]
        num_classes_pretrained = self.hparams["hparams"]["model"]["num_classes"]
        if "num_classes_pretrained" in self.hparams["hparams"]["model"]:
            num_classes_pretrained = self.hparams["hparams"]["model"]["num_classes_pretrained"]
        num_classes_init = num_classes_pretrained

        epistemic_version = "standard"
        if "epistemic_version" in self.hparams["hparams"]["model"]:
            epistemic_version = self.hparams["hparams"]["model"]["epistemic_version"]

        in_channels = self.hparams["hparams"]["model"]["in_channels"]
        dropout_prob = self.hparams["hparams"]["model"]["dropout_prob"]
        deep_encoder = self.hparams["hparams"]["model"]["deep_encoder"]

        self.num_mc_epistemic = self.hparams["hparams"]["train"]["num_mc_epistemic"]
        self.active_learning = "active_learning" in self.hparams["hparams"]
        self.num_train_data = num_train_data

        self.model = ERFNetModel(num_classes_init, in_channels, dropout_prob, deep_encoder, epistemic_version)

        self.inv_class_frequencies = None
        if "class_frequencies" in self.hparams["hparams"]["model"]:
            class_frequencies = torch.Tensor(self.hparams["hparams"]["model"]["class_frequencies"])
            self.inv_class_frequencies = class_frequencies.sum() / class_frequencies

    def replace_output_layer(self):
        self.model.decoder.output_conv = nn.ConvTranspose2d(
            16, self.num_classes, 2, stride=2, padding=0, output_padding=0, bias=True
        )

    def get_loss(self, x: torch.Tensor, y: torch.Tensor, device: torch.device = None):
        device = self.device if device is None else device
        if self.inv_class_frequencies is not None:
            self.inv_class_frequencies = self.inv_class_frequencies.to(device)

        loss = get_criterion(self.hparams["hparams"], self.inv_class_frequencies)
        return loss(x, y)

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return out

    def track_predictions(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        stage: str = "Train",
        step: int = 0,
        epistemic_uncertainties: np.array = None,
    ):
        sample_img_out = predictions[:1]
        sample_img_out = utils.toOneHot(sample_img_out, self.hparams["hparams"]["data"]["name"])
        self.logger.experiment.add_image(
            f"{stage}/Output image", torch.from_numpy(sample_img_out), step, dataformats="HWC"
        )

        sample_img_in = images[:1]
        grid = torchvision.utils.make_grid(sample_img_in)
        self.logger.experiment.add_image(f"{stage}/Input image", grid, step)

        sample_anno = targets[:1]
        sample_anno = utils.toOneHot(sample_anno, self.hparams["hparams"]["data"]["name"])
        self.logger.experiment.add_image(f"{stage}/Annotation", torch.from_numpy(sample_anno), step, dataformats="HWC")

        if epistemic_uncertainties is not None:
            sample_ep_uncertainty = epistemic_uncertainties.cpu().numpy()[0, :, :]
            sizes = sample_ep_uncertainty.shape
            fig = plt.figure()
            fig.set_size_inches(3.44 * sizes[0] / sizes[1], 3.44 * sizes[0] / sizes[1], forward=False)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            ax.imshow(sample_ep_uncertainty, cmap="plasma")
            fig.add_axes(ax)
            self.logger.experiment.add_figure(f"{stage}/Uncertainty/Epistemic", fig, step)

    def track_epistemic_uncertainty_stats(self, outputs: List[torch.Tensor]):
        per_class_ep_uncertainties = torch.stack([tmp["per_class_ep_uncertainty"] for tmp in outputs])
        per_class_ep_uncertainty = torch.mean(per_class_ep_uncertainties, dim=0)

        plt.figure(figsize=(10, 7))
        fig_ = sns.barplot(x=list(range(self.num_classes)), y=per_class_ep_uncertainty.tolist()).get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("UncertaintyStats/EpistemicPerClass", fig_, self.current_epoch)
        print(f"Mean per-class epistemic uncertainty: {per_class_ep_uncertainty}")

    def track_confusion_matrix(self, conf_matrices: List[torch.Tensor], stage: str = "Validation"):
        print("Confusion matrix:")
        total_conf_matrix = metrics.total_conf_matrix_from_conf_matrices(conf_matrices)
        df_cm = pd.DataFrame(
            total_conf_matrix.cpu().numpy(), index=range(self.num_classes), columns=range(self.num_classes)
        )
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure(f"ConfusionMatrix/{stage}", fig_, self.current_epoch)

    def training_step(self, batch: dict, batch_idx):
        out = self.forward(batch["data"])
        loss = self.get_loss(out, batch["anno"])
        _, preds = torch.max(out, dim=1)

        # self.track_predictions(batch["data"], preds, batch["anno"], stage="Train")

        self.log("train:loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx):
        out = self.forward(batch["data"])
        _, preds = torch.max(out, dim=1)
        loss = self.get_loss(out, batch["anno"])

        self.log("val:loss", loss, prog_bar=True)
        # self.track_predictions(batch["data"], preds, batch["anno"], stage="Validation")

        confusion_matrix = torchmetrics.functional.confusion_matrix(
            preds, batch["anno"], num_classes=self.num_classes, normalize=None
        )

        return {"conf_matrix": confusion_matrix}

    def validation_epoch_end(self, outputs):
        conf_matrices = [tmp["conf_matrix"] for tmp in outputs]
        iou = metrics.iou_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.hparams["hparams"]["data"]["name"]]
        )
        accuracy = metrics.accuracy_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.hparams["hparams"]["data"]["name"]]
        )

        self.log("val:acc", accuracy, prog_bar=True)
        self.log("val:iou", iou, prog_bar=True)
        self.track_confusion_matrix(conf_matrices, stage="Validation")

    def test_step(self, batch: dict, batch_idx):
        targets = batch["anno"]
        out = self.forward(batch["data"])

        loss = self.get_loss(out, targets)
        mean_prediction_probs, _, _, mutual_info_predictions = utils.get_mc_dropout_predictions(
            self, batch, self.num_mc_epistemic, aleatoric_model=False
        )
        mean_prediction_probs, mutual_info_predictions = torch.from_numpy(mean_prediction_probs).to(
            self.device
        ), torch.from_numpy(mutual_info_predictions).to(self.device)
        _, preds = torch.max(mean_prediction_probs, dim=1)

        confusion_matrix = torchmetrics.functional.confusion_matrix(
            preds, targets, num_classes=self.num_classes, normalize=None
        )

        calibration_info = metrics.compute_calibration_info(mean_prediction_probs, targets, num_bins=20)
        self.track_predictions(
            batch["data"], preds, targets, stage="Test", step=batch_idx, epistemic_uncertainties=mutual_info_predictions
        )

        per_class_ep_uncertainty = torch.zeros(self.num_classes)
        for predicted_class in torch.unique(preds):
            mask = preds == predicted_class
            per_class_ep_uncertainty[predicted_class.item()] = torch.mean(mutual_info_predictions[mask]).item()

        self.log("Test/Loss", loss, prog_bar=True)

        return {
            "conf_matrix": confusion_matrix,
            "per_class_ep_uncertainty": per_class_ep_uncertainty,
            "calibration_info": calibration_info,
        }

    def test_epoch_end(self, outputs):
        conf_matrices = [tmp["conf_matrix"] for tmp in outputs]
        calibration_info_list = [tmp["calibration_info"] for tmp in outputs]

        iou = metrics.iou_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.hparams["hparams"]["data"]["name"]]
        )
        accuracy = metrics.accuracy_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.hparams["hparams"]["data"]["name"]]
        )
        ece = metrics.ece_from_calibration_info(calibration_info_list, num_bins=20)

        self.log("Test/Acc", accuracy, prog_bar=True)
        self.log("Test/IoU", iou, prog_bar=True)
        self.log("Test/ECE", ece, prog_bar=True)

        fig_ = metrics.compute_calibration_plots(outputs)
        self.logger.experiment.add_figure("UncertaintyStats/Calibration", fig_, self.current_epoch)

        self.track_epistemic_uncertainty_stats(outputs)
        self.track_confusion_matrix(conf_matrices, stage="Test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["hparams"]["train"]["lr"],
            weight_decay=self.weight_decay,
        )
        return optimizer

    @property
    def weight_decay(self) -> float:
        if not self.active_learning:
            return self.hparams["hparams"]["train"]["weight_decay"]

        return (1 - self.hparams["hparams"]["model"]["dropout_prob"]) / (2 * self.num_train_data)


#######################################
# Modules                             #
#######################################


# Bayesian ERFNet
class BayesianERFNetModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        dropout_prob: float,
        use_shared_decoder: bool = False,
        deep_encoder: bool = False,
        epistemic_version: str = "standard",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_shared_decoder = use_shared_decoder

        self.encoder = DropoutERFNetEncoder(in_channels, dropout_prob, epistemic_version)
        if deep_encoder:
            self.encoder = DropoutERFNetDeepEncoder(in_channels, dropout_prob)

        self.segmentation_decoder = DropoutERFNetDecoder(self.num_classes)
        self.aleatoric_uncertainty_decoder = ERFNetAleatoricUncertaintyDecoder(self.num_classes)
        self.shared_decoder = ERFNetAleatoricSharedDecoder(self.num_classes, dropout_prob, epistemic_version)

    def forward(self, x):
        output_enc = self.encoder(x)

        if self.use_shared_decoder:
            output_seg, output_std = self.shared_decoder(output_enc)
        else:
            output_seg = self.segmentation_decoder(output_enc)
            output_std = self.aleatoric_uncertainty_decoder(output_enc)

        return output_seg, output_std


class DropoutERFNetEncoder(nn.Module):
    def __init__(self, in_channels: int, dropout_prob: float = 0.3, epistemic_version: str = "standard"):
        super().__init__()
        self.initial_block = blocks.DownsamplerBlock(in_channels, 16)

        self.layers = nn.ModuleList()

        self.layers.append(blocks.DownsamplerBlock(16, 64))

        dropout_prob_1, dropout_prob_2, dropout_prob_3 = self.get_dropout_probs(dropout_prob, epistemic_version)

        for x in range(0, 5):  # 5 times
            self.layers.append(blocks.non_bottleneck_1d(64, dropout_prob_1, 1))

        self.layers.append(blocks.DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            dropout_prob_tmp = dropout_prob_2 if x == 0 else dropout_prob_3
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 2))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 4))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 8))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 16))

    @staticmethod
    def get_dropout_probs(dropout_prob: float, epistemic_version: str) -> Tuple[float, float, float]:
        if epistemic_version == "all":
            return dropout_prob, dropout_prob, dropout_prob
        elif epistemic_version == "center":
            return 0, 0, dropout_prob
        elif epistemic_version == "classifier":
            return 0, 0, 0
        elif epistemic_version == "standard":
            return dropout_prob / 10, dropout_prob, dropout_prob
        else:
            raise ValueError(f"Epistemic version '{epistemic_version}' unknown!")

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class DropoutERFNetDeepEncoder(nn.Module):
    def __init__(self, in_channels: int, dropout_prob: float = 0.3):
        super().__init__()
        self.initial_block = blocks.DownsamplerBlock(in_channels, 16)

        self.layers = nn.ModuleList()

        self.layers.append(blocks.DownsamplerBlock(16, 64))

        for x in range(0, 10):  # 10 times
            self.layers.append(blocks.non_bottleneck_1d(64, dropout_prob / 10, 1))

        self.layers.append(blocks.DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 3 times
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob, 2))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob, 4))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob, 8))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob, 16))

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class DropoutERFNetDecoder(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(blocks.UpsamplerBlock(128, 64))
        self.layers.append(blocks.non_bottleneck_1d(64, 0, 1))
        self.layers.append(blocks.non_bottleneck_1d(64, 0, 1))

        self.layers.append(blocks.UpsamplerBlock(64, 16))
        self.layers.append(blocks.non_bottleneck_1d(16, 0, 1))
        self.layers.append(blocks.non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class ERFNetAleatoricSharedDecoder(nn.Module):
    def __init__(self, num_classes: int, dropout_prob: float = 0.0, epistemic_version: str = "standard"):
        super().__init__()

        self.num_classes = num_classes
        self.layers = nn.ModuleList()

        dropout_prob_1, dropout_prob_2, dropout_prob_3 = self.get_dropout_probs(dropout_prob, epistemic_version)

        self.layers.append(blocks.UpsamplerBlock(128, 64))
        self.layers.append(blocks.non_bottleneck_1d(64, dropout_prob_1, 1))
        self.layers.append(blocks.non_bottleneck_1d(64, dropout_prob_1, 1))

        self.layers.append(blocks.UpsamplerBlock(64, 16))
        self.layers.append(blocks.non_bottleneck_1d(16, dropout_prob_2, 1))
        self.layers.append(blocks.non_bottleneck_1d(16, dropout_prob_3, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes + 1, 2, stride=2, padding=0, output_padding=0, bias=True)
        self.output_fn = nn.Softplus(beta=1)

    @staticmethod
    def get_dropout_probs(dropout_prob: float, epistemic_version: str) -> Tuple[float, float, float]:
        if epistemic_version == "all":
            return dropout_prob, dropout_prob, dropout_prob
        elif epistemic_version == "center":
            return dropout_prob, 0, 0
        elif epistemic_version == "classifier":
            return 0, 0, dropout_prob
        elif epistemic_version == "standard":
            return 0, 0, 0
        else:
            raise ValueError(f"Epistemic version '{epistemic_version}' unknown!")

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = input

        for layer in self.layers:
            output = layer(output)

        output_seg, output_std = self.output_conv(output).split(self.num_classes, 1)
        output_std = self.output_fn(output_std) + 10 ** (-8)
        return output_seg, output_std


class ERFNetAleatoricUncertaintyDecoder(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.num_classes = num_classes
        self.layers = nn.ModuleList()

        self.layers.append(blocks.UpsamplerBlock(128, 64))
        self.layers.append(blocks.non_bottleneck_1d(64, 0, 1))
        self.layers.append(blocks.non_bottleneck_1d(64, 0, 1))

        self.layers.append(blocks.UpsamplerBlock(64, 16))
        self.layers.append(blocks.non_bottleneck_1d(16, 0, 1))
        self.layers.append(blocks.non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, 1, 2, stride=2, padding=0, output_padding=0, bias=True)
        self.output_fn = nn.Softplus(beta=1)

    def forward(self, x):
        output = x

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)
        output = self.output_fn(output) + 10 ** (-8)
        return output


# ERFNet
class ERFNetModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        dropout_prop: float = 0.0,
        deep_encoder: bool = False,
        epistemic_version: str = "standard",
    ):
        super().__init__()

        self.encoder = ERFNetEncoder(in_channels, dropout_prop, epistemic_version)
        if deep_encoder:
            self.encoder = ERFNetDeepEncoder(in_channels, dropout_prop)

        self.decoder = ERFNetDecoder(num_classes, dropout_prop, epistemic_version)

    def forward(self, input):
        output = self.encoder(input)
        return self.decoder(output)


class ERFNetEncoder(nn.Module):
    def __init__(self, in_channels: int, dropout_prob: float = 0.3, epistemic_version: str = "standard"):
        super().__init__()
        self.initial_block = blocks.DownsamplerBlock(in_channels, 16)

        self.layers = nn.ModuleList()

        self.layers.append(blocks.DownsamplerBlock(16, 64))

        dropout_prob_1, dropout_prob_2, dropout_prob_3 = self.get_dropout_probs(dropout_prob, epistemic_version)

        for x in range(0, 5):  # 5 times
            self.layers.append(blocks.non_bottleneck_1d(64, dropout_prob_1, 1))

        self.layers.append(blocks.DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            dropout_prob_tmp = dropout_prob_2 if x == 0 else dropout_prob_3
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 2))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 4))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 8))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 16))

    @staticmethod
    def get_dropout_probs(dropout_prob: float, epistemic_version: str) -> Tuple[float, float, float]:
        if epistemic_version == "all":
            return dropout_prob, dropout_prob, dropout_prob
        elif epistemic_version == "center":
            return 0, 0, dropout_prob
        elif epistemic_version == "classifier":
            return 0, 0, 0
        elif epistemic_version == "standard":
            return dropout_prob / 10, dropout_prob, dropout_prob
        else:
            raise ValueError(f"Epistemic version '{epistemic_version}' unknown!")

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class ERFNetDeepEncoder(nn.Module):
    def __init__(self, in_channels: int, dropout_prop: float = 0.3):
        super().__init__()
        self.initial_block = blocks.DownsamplerBlock(in_channels, 16)

        self.layers = nn.ModuleList()

        self.layers.append(blocks.DownsamplerBlock(16, 64))

        for x in range(0, 10):  # 10 times
            self.layers.append(blocks.non_bottleneck_1d(64, dropout_prop / 10, 1))

        self.layers.append(blocks.DownsamplerBlock(64, 128))

        for x in range(0, 3):  # 3 times
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prop, 2))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prop, 4))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prop, 8))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prop, 16))

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class ERFNetDecoder(nn.Module):
    def __init__(self, num_classes: int, dropout_prob: float = 0.0, epistemic_version: str = "standard"):
        super().__init__()

        self.layers = nn.ModuleList()

        dropout_prob_1, dropout_prob_2, dropout_prob_3 = self.get_dropout_probs(dropout_prob, epistemic_version)

        self.layers.append(blocks.UpsamplerBlock(128, 64))
        self.layers.append(blocks.non_bottleneck_1d(64, dropout_prob_1, 1))
        self.layers.append(blocks.non_bottleneck_1d(64, dropout_prob_1, 1))

        self.layers.append(blocks.UpsamplerBlock(64, 16))
        self.layers.append(blocks.non_bottleneck_1d(16, dropout_prob_2, 1))
        self.layers.append(blocks.non_bottleneck_1d(16, dropout_prob_3, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    @staticmethod
    def get_dropout_probs(dropout_prob: float, epistemic_version: str) -> Tuple[float, float, float]:
        if epistemic_version == "all":
            return dropout_prob, dropout_prob, dropout_prob
        elif epistemic_version == "center":
            return dropout_prob, 0, 0
        elif epistemic_version == "classifier":
            return 0, 0, dropout_prob
        elif epistemic_version == "standard":
            return 0, 0, 0
        else:
            raise ValueError(f"Epistemic version '{epistemic_version}' unknown!")

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output
