from typing import Dict, Tuple

import numpy as np
import torch
from agri_semantics.utils import resize
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torchvision import transforms

LABELS = {
    "scw": {
        "soil": {"color": (0, 0, 0), "id": 0},
        "crop": {"color": (0, 255, 0), "id": 1},
        "weed": {"color": (255, 0, 0), "id": 2},
    },
    "dlpLabels": {
        "soil": {"color": (0, 0, 0), "id": 0},
        "crop": {"color": (0, 255, 0), "id": 1},
        "weed": {"color": (255, 0, 0), "id": 2},
        "dicot": {"color": (0, 25, 127), "id": 3},
        "grass": {"color": (64, 127, 0), "id": 4},
        "vegetation": {"color": (0, 0, 255), "id": 5},
    },
    "stemLabels": {
        "soil": {"color": (0, 0, 0), "id": 0},
        "crop": {"color": (0, 255, 0), "id": 2},
        "dicot": {"color": (0, 0, 255), "id": 1},
    },
    "cityscapes": {
        "road": {"color": (128, 64, 128), "id": 0},
        "sidewalk": {"color": (244, 35, 232), "id": 1},
        "building": {"color": (70, 70, 70), "id": 2},
        "wall": {"color": (102, 102, 156), "id": 3},
        "fence": {"color": (190, 153, 153), "id": 4},
        "pole": {"color": (153, 153, 153), "id": 5},
        "traffic light": {"color": (250, 170, 30), "id": 6},
        "traffic sign": {"color": (220, 220, 0), "id": 7},
        "vegetation": {"color": (107, 142, 35), "id": 8},
        "terrain": {"color": (152, 251, 152), "id": 9},
        "sky": {"color": (70, 130, 180), "id": 10},
        "person": {"color": (220, 20, 60), "id": 11},
        "rider": {"color": (255, 0, 0), "id": 12},
        "car": {"color": (0, 0, 142), "id": 13},
        "truck": {"color": (0, 0, 70), "id": 14},
        "bus": {"color": (0, 60, 100), "id": 15},
        "train": {"color": (0, 80, 100), "id": 16},
        "motorcycle": {"color": (0, 0, 230), "id": 17},
        "bicycle": {"color": (119, 11, 32), "id": 18},
        "void": {"color": (0, 0, 0), "id": 19},
    },
    "rit18": {
        "bg": {"color": (0, 0, 0), "id": 0},
        "road marking": {"color": (19, 9, 25), "id": 1},
        "tree": {"color": (26, 24, 52), "id": 2},
        "building": {"color": (24, 45, 72), "id": 3},
        "vehicle": {"color": (21, 69, 78), "id": 4},
        "person": {"color": (25, 94, 70), "id": 5},
        "lifeguard chair": {"color": (43, 111, 57), "id": 6},
        "picnic table": {"color": (75, 120, 47), "id": 7},
        "black wood panel": {"color": (114, 122, 49), "id": 8},
        "white wood panel": {"color": (161, 121, 74), "id": 9},
        "landing pad": {"color": (193, 121, 111), "id": 10},
        "buoy": {"color": (209, 128, 156), "id": 11},
        "rocks": {"color": (211, 143, 197), "id": 12},
        "vegetation": {"color": (203, 165, 227), "id": 13},
        "grass": {"color": (194, 193, 242), "id": 14},
        "sand": {"color": (194, 216, 242), "id": 15},
        "lake": {"color": (206, 235, 239), "id": 16},
        "pond": {"color": (229, 247, 240), "id": 17},
        "asphalt": {"color": (255, 255, 255), "id": 18},
    },
    "potsdam": {
        "boundary line": {"color": (0, 0, 0), "id": 0},
        "imprevious surfaces": {"color": (255, 255, 255), "id": 1},
        "building": {"color": (0, 0, 255), "id": 2},
        "low vegetation": {"color": (0, 255, 255), "id": 3},
        "tree": {"color": (0, 255, 0), "id": 4},
        "car": {"color": (255, 255, 0), "id": 5},
        "clutter/background": {"color": (255, 0, 0), "id": 6},
    },
    "flightmare": {
        "background": {"color": (0, 0, 0), "id": 0},
        "floor": {"color": (2, 73, 9), "id": 1},
        "hangar": {"color": (32, 73, 65), "id": 2},
        "fence": {"color": (6, 73, 72), "id": 3},
        "road": {"color": (36, 73, 8), "id": 4},
        "tank": {"color": (2, 73, 128), "id": 5},
        "pipe": {"color": (32, 9, 201), "id": 6},
        "container": {"color": (6, 9, 193), "id": 7},
        "misc": {"color": (36, 9, 129), "id": 8},
        "boundary": {"color": (255, 255, 255), "id": 9},
    },
}

THEMES = {
    "cityscapes": "cityscapes",
    "weedmap": "scw",
    "rit18": "rit18",
    "potsdam": "potsdam",
    "flightmare": "flightmare",
}


def imap2rgb(imap, channel_order, theme):
    """converts an iMap label image into a RGB Color label image,
    following label colors/ids stated in the "labels" dict.

    Arguments:
        imap {numpy with shape (h,w)} -- label image containing label ids [int]
        channel_order {str} -- channel order ['hwc' for shape(h,w,3) or 'chw' for shape(3,h,w)]
        theme {str} -- label theme

    Returns:
        float32 numpy with shape (channel_order) -- rgb label image containing label colors from dict (int,int,int)
    """
    assert channel_order == "hwc" or channel_order == "chw"
    assert len(imap.shape) == 2
    assert theme in LABELS.keys()

    rgb = np.zeros((imap.shape[0], imap.shape[1], 3), np.float32)
    for _, cl in LABELS[theme].items():  # loop each class label
        if cl["color"] == (0, 0, 0):
            continue  # skip assignment of only zeros
        mask = imap == cl["id"]
        rgb[:, :, 0][mask] = cl["color"][0]
        rgb[:, :, 1][mask] = cl["color"][1]
        rgb[:, :, 2][mask] = cl["color"][2]
    if channel_order == "chw":
        rgb = np.moveaxis(rgb, -1, 0)  # convert hwc to chw
    return rgb


def toOneHot(tensor, dataset_name):
    img = tensor.detach().cpu().numpy()[0]
    if len(img.shape) == 3:
        img = np.transpose(img, (1, 2, 0))
        img = np.argmax(img, axis=-1)

    img = imap2rgb(img, channel_order="hwc", theme=THEMES[dataset_name])
    return img.astype(np.uint8)


def enable_dropout(model: nn.Module):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def sample_from_aleatoric_model(
    model: LightningModule, batch: Dict, num_mc_aleatoric: int = 50, device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    est_seg, est_std, hidden_representation = model.forward(batch["data"])
    sampled_predictions = torch.zeros((num_mc_aleatoric, *est_seg.size()), device=device)
    for j in range(num_mc_aleatoric):
        noise_mean = torch.zeros(est_seg.size(), device=device)
        noise_std = torch.ones(est_seg.size(), device=device)
        epsilon = torch.distributions.normal.Normal(noise_mean, noise_std).sample()
        sampled_seg = est_seg + torch.mul(est_std, epsilon)
        sampled_predictions[j] = sampled_seg
    return torch.mean(sampled_predictions, dim=0), hidden_representation


def compute_prediction_stats(
    predictions: np.array, hidden_representations: np.array
) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    mean_predictions = np.mean(predictions, axis=0)

    variance_predictions = np.var(predictions, axis=0)
    if variance_predictions.shape[1] == 1:
        variance_predictions = variance_predictions.squeeze(axis=1)

    entropy_predictions = -np.sum(mean_predictions * np.log(mean_predictions + 10 ** (-8)), axis=1)
    mutual_info_predictions = entropy_predictions - np.mean(
        np.sum(-predictions * np.log(predictions + 10 ** (-8)), axis=2), axis=0
    )

    hidden_representations = np.mean(hidden_representations, axis=0)

    return mean_predictions, variance_predictions, entropy_predictions, mutual_info_predictions, hidden_representations


def get_predictions(
    model: LightningModule,
    batch: Dict,
    num_mc_dropout: int = 50,
    aleatoric_model: bool = True,
    num_mc_aleatoric: int = 50,
    ensemble_model: bool = False,
    device: torch.device = None,
    task: str = "classification",
) -> Tuple[np.array, np.array, np.array]:
    use_mc_dropout = num_mc_dropout > 1 and not ensemble_model
    num_mc_dropout = num_mc_dropout if num_mc_dropout > 1 else 1

    num_predictions = num_mc_dropout
    if ensemble_model:
        num_predictions = len(model.models)

    softmax = nn.Softmax(dim=1)
    predictions = []
    hidden_representations = []

    for i in range(num_predictions):
        if ensemble_model:
            single_model = model.models[i].to(device)
        else:
            single_model = model.to(device)

        single_model.eval()
        if use_mc_dropout:
            enable_dropout(single_model)

        with torch.no_grad():
            if aleatoric_model:
                est_anno, hidden_representation = sample_from_aleatoric_model(
                    single_model, batch, num_mc_aleatoric=num_mc_aleatoric, device=device
                )
            else:
                est_anno, hidden_representation = single_model.forward(batch["data"])

            if task == "classification":
                est_seg_probs = softmax(est_anno)
                predictions.append(est_seg_probs.cpu().numpy())
            elif task == "regression":
                predictions.append(est_anno.cpu().numpy())
            else:
                raise NotImplementedError(f"{task} output non-linearity not implemented!")

            hidden_representations.append(hidden_representation.cpu().numpy())

    (
        mean_predictions,
        variance_predictions,
        entropy_predictions,
        mutual_info_predictions,
        hidden_representations,
    ) = compute_prediction_stats(np.array(predictions), np.array(hidden_representations))

    if task == "regression":
        uncertainty_predictions = variance_predictions
    elif task == "classification":
        uncertainty_predictions = mutual_info_predictions if use_mc_dropout or ensemble_model else entropy_predictions
    else:
        raise NotImplementedError(f"Uncertainty measure for {task} task not implemented!")

    return mean_predictions, uncertainty_predictions, hidden_representations


def infer_anno_and_epistemic_uncertainty_from_image(
    model: LightningModule,
    image: np.array,
    num_mc_epistemic: int = 25,
    resize_image: bool = False,
    aleatoric_model: bool = True,
    num_mc_aleatoric: int = 50,
    ensemble_model: bool = False,
    task: str = "classification",
) -> Tuple[np.array, np.array, np.array]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_normalized_tensor = transforms.ToTensor()
    image_tensor = to_normalized_tensor(image)

    if resize_image:
        image_tensor = resize(image_tensor, width=344, height=None, interpolation=1, keep_aspect_ratio=True)

    image_batch = {"data": image_tensor.float().unsqueeze(0).to(device)}
    mean_predictions, uncertainty_predictions, hidden_representations = get_predictions(
        model,
        image_batch,
        num_mc_dropout=num_mc_epistemic,
        aleatoric_model=aleatoric_model,
        num_mc_aleatoric=num_mc_aleatoric,
        ensemble_model=ensemble_model,
        device=device,
        task=task,
    )

    return (
        np.squeeze(mean_predictions, axis=0),
        np.squeeze(uncertainty_predictions, axis=0),
        np.squeeze(hidden_representations, axis=0),
    )
