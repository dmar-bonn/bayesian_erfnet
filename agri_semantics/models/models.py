import os
import sys
from typing import Dict, List, Optional, Tuple

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

import agri_semantics.utils.utils as utils
from agri_semantics.constants import Losses
from agri_semantics.models.loss import CrossEntropyLoss
from agri_semantics.utils import metrics
from agri_semantics.models import modules


##############################################################################################
#                                                                                            #
#  Pytorch Lightning ERFNet implementation from Jan Weyler. Our Bayesian-ERFNet              #
#  implementation builds upon Jan's ERFNet implementation.                                   #
#                                                                                            #
##############################################################################################

IGNORE_INDEX = {"cityscapes": 19, "weedmap": -100, "rit18": 0, "rit18_merged": 0, "potsdam": 0, "flightmare": 9}


class NetworkWrapper(LightningModule):
    def __init__(self, cfg: Dict, al_logger_name: str = "", al_iteration: int = 0, num_train_data: int = 1):
        super().__init__()

        self.cfg = cfg

        self.task = cfg["model"]["task"]
        self.num_classes = self.cfg["model"]["num_classes"]
        num_classes_pretrained = self.cfg["model"]["num_classes"]
        if "num_classes_pretrained" in self.cfg["model"]:
            num_classes_pretrained = self.cfg["model"]["num_classes_pretrained"]
        self.num_classes_init = num_classes_pretrained

        self.epistemic_version = "standard"
        if "epistemic_version" in self.cfg["model"]:
            self.epistemic_version = self.cfg["model"]["epistemic_version"]

        self.in_channels = self.cfg["model"]["in_channels"]
        self.dropout_prob = self.cfg["model"]["dropout_prob"]
        self.deep_encoder = self.cfg["model"]["deep_encoder"]

        self.ensemble_model = self.cfg["model"]["ensemble_model"]
        self.aleatoric_model = self.cfg["model"]["aleatoric_model"]

        self.num_mc_aleatoric = self.cfg["train"]["num_mc_aleatoric"]
        self.num_mc_epistemic = self.cfg["train"]["num_mc_epistemic"]

        self.active_learning = "active_learning" in self.cfg
        self.al_logger_name = al_logger_name
        self.al_iteration = al_iteration
        self.num_train_data = num_train_data
        self.test_evaluation_metrics = {}

        self.inv_class_frequencies = None
        if "class_frequencies" in self.cfg["model"]:
            class_frequencies = torch.Tensor(self.cfg["model"]["class_frequencies"])
            self.inv_class_frequencies = class_frequencies.sum() / class_frequencies
            self.inv_class_frequencies = self.inv_class_frequencies.to(self.device)

    @staticmethod
    def identity_output_fn(x: torch.Tensor) -> torch.Tensor:
        identity_fn = nn.Identity()
        return identity_fn(x)

    def regression_output_fn(self, x: torch.Tensor) -> torch.Tensor:
        min_value = self.cfg["model"]["value_range"]["min_value"]
        max_value = self.cfg["model"]["value_range"]["max_value"]
        value_range = max_value - min_value
        sigmoid_fn = nn.Sigmoid()
        return sigmoid_fn(x) * value_range - min_value

    @property
    def output_fn(self) -> Optional[callable]:
        if self.task == "classification":
            return self.identity_output_fn
        elif self.task == "regression":
            return self.regression_output_fn
        else:
            raise NotImplementedError(f"{self.task} output non-linearity not implemented!")

    @property
    def loss_fn(self) -> callable:
        loss_name = self.cfg["model"]["loss"]
        if loss_name == Losses.CROSS_ENTROPY:
            return CrossEntropyLoss(
                ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]],
                weight=self.inv_class_frequencies,
            )
        elif loss_name == Losses.MSE:
            return nn.MSELoss(reduction="mean")
        else:
            raise RuntimeError(f"Loss {loss_name} not available!")

    def replace_output_layer(self):
        pass

    def training_step(self, batch: Dict, batch_idx: int):
        pass

    def validation_step(self, batch: dict, batch_idx: int):
        pass

    def validation_epoch_end(self, outputs):
        conf_matrices = [tmp["conf_matrix"] for tmp in outputs]
        losses = [tmp["loss"] for tmp in outputs]
        self.track_evaluation_metrics(conf_matrices, losses, stage="Validation", calibration_info_list=None)
        if self.task == "classification":
            self.track_confusion_matrix(conf_matrices, stage="Validation")

    def test_step(self, batch: dict, batch_idx: int):
        pass

    def common_test_step(self, batch: dict, batch_idx: int, aleatoric_dist: Tuple = None):
        targets = batch["anno"].to(self.device)
        mean_predictions, uncertainty_predictions, hidden_representations = utils.get_predictions(
            self,
            batch,
            num_mc_dropout=self.num_mc_epistemic,
            aleatoric_model=self.aleatoric_model,
            num_mc_aleatoric=self.num_mc_aleatoric,
            ensemble_model=self.ensemble_model,
            device=self.device,
            task=self.task,
        )
        mean_predictions, uncertainty_predictions = torch.from_numpy(mean_predictions).to(
            self.device
        ), torch.from_numpy(uncertainty_predictions).to(self.device)

        if self.task == "classification":
            _, hard_preds = torch.max(mean_predictions, dim=1)
        else:
            hard_preds = mean_predictions

        loss = self.get_loss(mean_predictions, targets)
        self.log("Test/Loss", loss, prog_bar=True)

        confusion_matrix = None
        calibration_info = None
        per_class_epistemic_uncertainty = None
        if self.task == "classification":
            confusion_matrix = torchmetrics.functional.confusion_matrix(
                hard_preds, targets, num_classes=self.num_classes, normalize=None
            )
            calibration_info = metrics.compute_calibration_info(mean_predictions, targets, num_bins=20)
            per_class_epistemic_uncertainty = self.compute_per_class_epistemic_uncertainty(
                hard_preds, uncertainty_predictions
            )

        self.track_predictions(
            batch["data"],
            hard_preds,
            mean_predictions,
            targets,
            stage="Test",
            step=batch_idx,
            epistemic_uncertainties=uncertainty_predictions,
            dist=aleatoric_dist,
        )

        return {
            "conf_matrix": confusion_matrix,
            "loss": loss,
            "per_class_ep_uncertainty": per_class_epistemic_uncertainty,
            "calibration_info": calibration_info,
        }

    def test_epoch_end(self, outputs):
        conf_matrices = [tmp["conf_matrix"] for tmp in outputs]
        losses = [tmp["loss"] for tmp in outputs]
        calibration_info_list = [tmp["calibration_info"] for tmp in outputs]

        self.test_evaluation_metrics = self.track_evaluation_metrics(
            conf_matrices, losses, stage="Test", calibration_info_list=calibration_info_list
        )

        if self.task == "classification":
            self.track_confusion_matrix(conf_matrices, stage="Test")
            self.track_epistemic_uncertainty_stats(outputs, stage="Test")

            fig_ = metrics.compute_calibration_plots(outputs)
            self.logger.experiment.add_figure("UncertaintyStats/Calibration", fig_, self.current_epoch)

    def track_classification_metrics(
        self, conf_matrices: List, stage: str = "Test", calibration_info_list: List = None
    ) -> Dict:
        miou = metrics.mean_iou_from_conf_matrices(conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]])
        per_class_iou = metrics.per_class_iou_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )
        accuracy = metrics.accuracy_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )
        precision = metrics.precision_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )
        recall = metrics.recall_from_conf_matrices(conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]])
        f1_score = metrics.f1_score_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )

        ece = -1.0
        if calibration_info_list is not None:
            ece = metrics.ece_from_calibration_info(calibration_info_list, num_bins=20)

        self.log(f"{stage}/Precision", precision)
        self.log(f"{stage}/Recall", recall)
        self.log(f"{stage}/F1-Score", f1_score)
        self.log(f"{stage}/Acc", accuracy)
        self.log(f"{stage}/mIoU", miou)
        self.log(f"{stage}/ECE", ece)

        return {
            f"{stage}/Precision": precision,
            f"{stage}/Recall": recall,
            f"{stage}/F1-Score": f1_score,
            f"{stage}/Acc": accuracy,
            f"{stage}/mIoU": miou,
            f"{stage}/Per-Class-IoU": per_class_iou.tolist(),
            f"{stage}/ECE": ece,
        }

    def track_regression_metrics(self, mse_losses: List, stage: str = "Test") -> Dict:
        mse_losses = torch.Tensor([mse_loss.item() for mse_loss in mse_losses])
        mse = torch.mean(mse_losses).item()
        rmse = torch.sqrt(torch.mean(mse_losses)).item()

        self.log(f"{stage}/MSE", mse)
        self.log(f"{stage}/RMSE", rmse)

        return {f"{stage}/MSE": mse, f"{stage}/RMSE": rmse}

    def track_evaluation_metrics(
        self, conf_matrices: List, losses: List, stage: str = "Test", calibration_info_list: List = None
    ) -> Dict:
        if self.task == "classification":
            return self.track_classification_metrics(
                conf_matrices, stage=stage, calibration_info_list=calibration_info_list
            )
        else:
            return self.track_regression_metrics(losses, stage=stage)

    def track_epistemic_uncertainty_stats(self, outputs: List[torch.Tensor], stage: str = "Validation"):
        per_class_ep_uncertainties = torch.stack([tmp["per_class_ep_uncertainty"] for tmp in outputs])
        per_class_ep_uncertainty = torch.mean(per_class_ep_uncertainties, dim=0)

        ax = sns.barplot(x=list(range(self.num_classes)), y=per_class_ep_uncertainty.tolist())
        ax.set_xlabel("Class Index")
        ax.set_ylabel("Model Uncertainty [0,1]")

        if stage == "Test" and self.active_learning:
            plt.savefig(
                os.path.join(self.al_logger_name, f"per_class_ep_uncertainty_{self.al_iteration}.png"),
                dpi=300,
            )

        self.logger.experiment.add_figure(
            f"UncertaintyStats/{stage}/EpistemicPerClass", ax.get_figure(), self.current_epoch
        )

        plt.close()
        plt.clf()
        plt.cla()

    def track_confusion_matrix(self, conf_matrices: List[torch.Tensor], stage: str = "Validation"):
        total_conf_matrix = metrics.total_conf_matrix_from_conf_matrices(conf_matrices)
        df_cm = pd.DataFrame(
            total_conf_matrix.cpu().numpy(), index=range(self.num_classes), columns=range(self.num_classes)
        )

        ax = sns.heatmap(df_cm, annot=True, cmap="Spectral")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground Truth")

        if stage == "Test" and self.active_learning:
            plt.savefig(
                os.path.join(self.al_logger_name, f"confusion_matrix_{self.al_iteration}.png"),
                dpi=300,
            )

        self.logger.experiment.add_figure(f"ConfusionMatrix/{stage}", ax.get_figure(), self.current_epoch)

        plt.close()
        plt.clf()
        plt.cla()

    def track_gradient_norms(self):
        total_grad_norm = 0
        for params in self.model.parameters():
            if params.grad is not None:
                total_grad_norm += params.grad.data.norm(2).item()

        self.log(f"LossStats/GradientNorm", total_grad_norm)

    def track_predictions(
        self,
        images: torch.Tensor,
        hard_predictions: torch.Tensor,
        prob_predictions: torch.Tensor,
        targets: torch.Tensor,
        stage: str = "Train",
        step: int = 0,
        epistemic_uncertainties: np.array = None,
        dist: Tuple[torch.Tensor, torch.Tensor] = None,
    ):
        sample_img_out = hard_predictions[:1]
        if self.task == "classification":
            sample_img_out = utils.toOneHot(sample_img_out, self.cfg["data"]["name"])
            self.logger.experiment.add_image(
                f"{stage}/Output image", torch.from_numpy(sample_img_out), step, dataformats="HWC"
            )
        else:
            sample_img_out = sample_img_out.cpu().numpy()[0, 0, :, :]
            sizes = sample_img_out.shape
            fig = plt.figure()
            fig.set_size_inches(3.44 * sizes[0] / sizes[1], 3.44 * sizes[0] / sizes[1], forward=False)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            ax.imshow(sample_img_out, cmap="gray")
            fig.add_axes(ax)
            self.logger.experiment.add_figure(f"{stage}/Output image", fig, step)

        sample_img_in = images[:1]
        sample_anno = targets[:1]

        self.logger.experiment.add_image(f"{stage}/Input image", sample_img_in.squeeze(), step, dataformats="CHW")

        if self.task == "classification":
            sample_prob_prediction = prob_predictions[:1]
            cross_entropy_fn = CrossEntropyLoss(reduction="none")
            sample_error_img = cross_entropy_fn(sample_prob_prediction, sample_anno).squeeze()
        else:
            sample_error_img = sample_img_out - sample_anno

        sizes = sample_img_out.shape
        fig = plt.figure()
        fig.set_size_inches(3.44 * sizes[0] / sizes[1], 3.44 * sizes[0] / sizes[1], forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        ax.imshow(sample_error_img.cpu().numpy(), cmap="gray")
        fig.add_axes(ax)
        self.logger.experiment.add_figure(f"{stage}/Error image", fig, step)

        if self.task == "classification":
            sample_anno = utils.toOneHot(sample_anno, self.cfg["data"]["name"])
            self.logger.experiment.add_image(
                f"{stage}/Annotation", torch.from_numpy(sample_anno), step, dataformats="HWC"
            )
        else:
            sample_anno = sample_anno.cpu().numpy()[0, 0, :, :]
            sizes = sample_anno.shape
            fig = plt.figure()
            fig.set_size_inches(3.44 * sizes[0] / sizes[1], 3.44 * sizes[0] / sizes[1], forward=False)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            ax.imshow(sample_anno, cmap="gray")
            fig.add_axes(ax)
            self.logger.experiment.add_figure(f"{stage}/Annotation", fig, step)

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

        if dist is not None:
            sample_aleatoric_unc_out = self.compute_aleatoric_uncertainty(dist[0], dist[1])[0, :, :]
            fig = plt.figure()
            plt.axis("off")
            plt.imshow(sample_aleatoric_unc_out, cmap="plasma")
            self.logger.experiment.add_figure(f"{stage}/Uncertainty/Aleatoric", fig, step)

    def compute_aleatoric_uncertainty(self, seg: torch.Tensor, std: torch.Tensor):
        pass

    def compute_per_class_epistemic_uncertainty(
        self, preds: torch.Tensor, uncertainty_predictions: torch.Tensor
    ) -> torch.Tensor:
        per_class_ep_uncertainty = torch.zeros(self.num_classes)
        for predicted_class in torch.unique(preds):
            mask = preds == predicted_class
            per_class_ep_uncertainty[predicted_class.item()] = torch.mean(uncertainty_predictions[mask]).item()

        return per_class_ep_uncertainty

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg["train"]["lr"],
            weight_decay=self.weight_decay,
        )
        return optimizer

    @property
    def weight_decay(self) -> float:
        if not self.active_learning:
            return self.cfg["train"]["weight_decay"]

        return (1 - self.cfg["model"]["dropout_prob"]) / (2 * self.num_train_data)


class AleatoricERFNet(NetworkWrapper):
    def __init__(self, cfg: Dict, al_logger_name: str = "", al_iteration: int = 0, num_train_data: int = 1):
        super(AleatoricERFNet, self).__init__(
            cfg, al_logger_name=al_logger_name, al_iteration=al_iteration, num_train_data=num_train_data
        )
        self.save_hyperparameters()

        self.model = modules.AleatoricERFNetModel(
            self.num_classes_init,
            self.in_channels,
            self.dropout_prob,
            use_shared_decoder=self.shared_decoder,
            deep_encoder=self.deep_encoder,
            epistemic_version=self.epistemic_version,
            output_fn=self.output_fn,
        )

    def replace_output_layer(self):
        if self.model.use_shared_decoder:
            self.model.shared_decoder.output_conv = nn.ConvTranspose2d(
                16, self.num_classes + 1, 2, stride=2, padding=0, output_padding=0, bias=True
            )
        else:
            self.model.segmentation_decoder.output_conv = nn.ConvTranspose2d(
                16, self.num_classes, 2, stride=2, padding=0, output_padding=0, bias=True
            )

    def get_loss(self, seg: torch.Tensor, std: torch.Tensor, true_seg: torch.Tensor) -> torch.Tensor:
        sampled_predictions = torch.zeros((self.num_mc_aleatoric, *seg.size()), device=self.device)

        for i in range(self.num_mc_aleatoric):
            noise_mean = torch.zeros(seg.size(), device=self.device)
            noise_std = torch.ones(seg.size(), device=self.device)
            epsilon = torch.distributions.normal.Normal(noise_mean, noise_std).sample()
            sampled_seg = seg + torch.mul(std, epsilon)
            sampled_predictions[i] = sampled_seg

        mean_prediction = torch.mean(sampled_predictions, dim=0)
        return self.loss_fn(mean_prediction, true_seg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_seg, output_std, hidden_representation = self.model(x)
        return output_seg, output_std, hidden_representation

    def compute_aleatoric_uncertainty(self, seg: torch.Tensor, std: torch.Tensor) -> np.array:
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

    def track_aleatoric_stats(self, std: torch.Tensor):
        self.log("Variance/TrainMin", torch.min(std))
        self.log("Variance/TrainMax", torch.max(std))
        self.log("Variance/TrainMean", torch.mean(std))

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        est_seg, est_std, _ = self.forward(batch["data"])
        loss = self.get_loss(est_seg, est_std, batch["anno"])

        self.track_aleatoric_stats(est_std)
        self.track_gradient_norms()
        self.log("train:loss", loss)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Dict:
        est_seg, est_std, _ = self.forward(batch["data"])
        loss = self.get_loss(est_seg, est_std, batch["anno"])
        _, preds = torch.max(est_seg, dim=1)

        self.log("Validation/Loss", loss, prog_bar=True)
        self.track_aleatoric_stats(est_std)

        confusion_matrix = None
        if self.task == "classification":
            _, preds = torch.max(est_seg, dim=1)
            confusion_matrix = torchmetrics.functional.confusion_matrix(
                preds, batch["anno"], num_classes=self.num_classes, normalize=None
            )

        return {"conf_matrix": confusion_matrix, "loss": loss}

    def test_step(self, batch: dict, batch_idx: int) -> Dict:
        est_seg, est_std, _ = self.forward(batch["data"])

        return self.common_test_step(batch, batch_idx, (est_seg, est_std))


class ERFNet(NetworkWrapper):
    def __init__(self, cfg: Dict, al_logger_name: str = "", al_iteration: int = 0, num_train_data: int = 1):
        super(ERFNet, self).__init__(
            cfg, al_logger_name=al_logger_name, al_iteration=al_iteration, num_train_data=num_train_data
        )

        self.model = modules.ERFNetModel(
            self.num_classes_init,
            self.in_channels,
            dropout_prop=self.dropout_prob,
            deep_encoder=self.deep_encoder,
            epistemic_version=self.epistemic_version,
            output_fn=self.output_fn,
        )

    def replace_output_layer(self):
        self.model.decoder.output_conv = nn.ConvTranspose2d(
            16, self.num_classes, 2, stride=2, padding=0, output_padding=0, bias=True
        )

    def get_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(x, y)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, hidden_representation = self.model(x)
        return out, hidden_representation

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out, _ = self.forward(batch["data"])
        loss = self.get_loss(out, batch["anno"])

        self.log("train:loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Dict:
        out, _ = self.forward(batch["data"])
        loss = self.get_loss(out, batch["anno"])
        self.log("Validation/Loss", loss, prog_bar=True)

        confusion_matrix = None
        if self.task == "classification":
            _, preds = torch.max(out, dim=1)
            confusion_matrix = torchmetrics.functional.confusion_matrix(
                preds, batch["anno"], num_classes=self.num_classes, normalize=None
            )

        return {"conf_matrix": confusion_matrix, "loss": loss}

    def test_step(self, batch: dict, batch_idx: int) -> Dict:
        return self.common_test_step(batch, batch_idx, aleatoric_dist=None)


class EnsembleNet(NetworkWrapper):
    def __init__(self, cfg: Dict, models: List[nn.Module], al_logger_name: str = "", al_iteration: int = 0):
        super(EnsembleNet, self).__init__(cfg, al_logger_name=al_logger_name, al_iteration=al_iteration)

        self.models = models

    def get_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(x, y)

    def test_step(self, batch: dict, batch_idx: int) -> Dict:
        return self.common_test_step(batch, batch_idx, aleatoric_dist=None)
