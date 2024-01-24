import copy
from typing import Dict

import numpy as np
import torch
from agri_semantics.datasets import get_data_module
from agri_semantics.models import get_model
from agri_semantics.utils import utils
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class ActiveLearner:
    def __init__(self, cfg: Dict, weights_path: str, checkpoint_path: str, strategy_name: str):
        self.cfg = cfg
        self.weights_path = weights_path
        self.initial_checkpoint_path = checkpoint_path
        self.strategy_name = strategy_name
        self.patience = cfg["train"]["patience"]
        self.test_statistics = {}

        self.data_module = self.setup_data_module()
        self.model = self.setup_model()
        self.trainer = self.setup_trainer(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_collected_images = min(
            cfg["active_learning"]["num_collected_images"], len(self.data_module.unlabeled_dataloader().dataset)
        )
        self.max_collected_images = min(
            cfg["active_learning"]["max_collected_images"], len(self.data_module.unlabeled_dataloader().dataset)
        )

    def setup_data_module(self) -> LightningDataModule:
        data_module = get_data_module(self.cfg)
        data_module.setup()

        return data_module

    def setup_model(self, num_train_data: int = 1) -> LightningModule:
        model = get_model(self.cfg, num_train_data=num_train_data)
        if self.weights_path:
            model = model.load_from_checkpoint(self.weights_path, hparams=self.cfg)
            if self.cfg["model"]["num_classes_pretrained"] != self.cfg["model"]["num_classes"]:
                model.replace_output_layer()

        return model

    def setup_trainer(self, iter_count: int):
        early_stopping = EarlyStopping(
            monitor="Validation/mIoU", min_delta=0.00, patience=self.patience, verbose=False, mode="max"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_saver = ModelCheckpoint(
            monitor="Validation/mIoU",
            filename=f"{self.cfg['experiment']['id']}_{self.strategy_name}_iter{str(iter_count)}"
            + "_{epoch:02d}_{iou:.2f}",
            mode="max",
            save_last=True,
        )
        tb_logger = pl_loggers.TensorBoardLogger(
            f"experiments/{self.cfg['experiment']['id']}",
            name=self.strategy_name,
            version=iter_count,
            default_hp_metric=False,
        )

        trainer = Trainer(
            accelerator="gpu",
            devices=self.cfg["train"]["n_gpus"],
            logger=tb_logger,
            resume_from_checkpoint=self.initial_checkpoint_path,
            max_epochs=self.cfg["train"]["max_epoch"],
            callbacks=[lr_monitor, checkpoint_saver, early_stopping],
            log_every_n_steps=1,
        )

        return trainer

    def select_data(self):
        raise NotImplementedError(f"select_data function of {self.strategy_name} active learner not implemented!")

    def update_train_data(self):
        self.data_module.append_data_indices(self.select_data())

    def retrain_model(self, num_train_data: int):
        self.model = self.setup_model(num_train_data=num_train_data)
        self.trainer.fit(self.model, self.data_module)
        self.model = self.model.load_from_checkpoint(
            self.trainer.checkpoint_callback.best_model_path, hparams=self.cfg_fine_tuned
        )

    @property
    def cfg_fine_tuned(self) -> Dict:
        cft_fine_tuned = copy.deepcopy(self.cfg)
        cft_fine_tuned["model"]["num_classes_pretrained"] = cft_fine_tuned["model"]["num_classes"]
        return cft_fine_tuned

    def plot_test_statistics(self, test_statistics: Dict):
        num_train_data = len(self.data_module.train_dataloader().dataset)
        self.test_statistics[num_train_data] = test_statistics
        self.model.logger.experiment.add_scalar("ActiveLearning/Loss", test_statistics["Test/Loss"], num_train_data)
        self.model.logger.experiment.add_scalar("ActiveLearning/Acc", test_statistics["Test/Acc"], num_train_data)
        self.model.logger.experiment.add_scalar("ActiveLearning/F1", test_statistics["Test/F1"], num_train_data)
        self.model.logger.experiment.add_scalar("ActiveLearning/mIoU", test_statistics["Test/mIoU"], num_train_data)
        self.model.logger.experiment.add_scalar("ActiveLearning/ECE", test_statistics["Test/ECE"], num_train_data)

    def run(self):
        iter_count = 0
        print(f"START {self.strategy_name} ACTIVE LEARNER RUN")
        while len(self.data_module.train_dataloader().dataset) < self.max_collected_images:
            self.update_train_data()
            print(f"TRAIN DATA: {len(self.data_module.train_dataloader().dataset)} of {self.max_collected_images}")

            self.trainer = self.setup_trainer(iter_count)
            self.retrain_model(len(self.data_module.train_dataloader().dataset))

            test_results = self.trainer.test(self.model, self.data_module)[0]
            self.plot_test_statistics(test_results)
            iter_count += 1


class RandomActiveLearner(ActiveLearner):
    def __init__(self, cfg: Dict, weights_path: str, checkpoint_path: str, strategy_name: str):
        super(RandomActiveLearner, self).__init__(cfg, weights_path, checkpoint_path, strategy_name)

    def select_data(self):
        unlabeled_indices = self.data_module.get_unlabeled_data_indices()
        sample_size = min(self.num_collected_images, len(unlabeled_indices))
        return np.random.choice(unlabeled_indices, size=sample_size, replace=False)


class BALDActiveLearner(ActiveLearner):
    def __init__(self, cfg: Dict, weights_path: str, checkpoint_path: str, strategy_name: str):
        super(BALDActiveLearner, self).__init__(cfg, weights_path, checkpoint_path, strategy_name)

        self.num_mc_epistemic = cfg["train"]["num_mc_epistemic"]
        self.num_mc_aleatoric = cfg["train"]["num_mc_aleatoric"]
        self.aleatoric_model = cfg["model"]["aleatoric_model"]
        self.evidential_model = cfg["model"]["evidential_model"]

    def select_data(self):
        unlabeled_indices = self.data_module.get_unlabeled_data_indices()
        sample_size = min(self.num_collected_images, len(unlabeled_indices))
        unlabeled_dataloader = self.data_module.unlabeled_dataloader()
        bald_objectives = np.ones(len(unlabeled_indices))

        for j, batch in enumerate(unlabeled_dataloader):
            (
                mean_predictions,
                epistemic_unc_predictions,
                aleatoric_unc_predictions,
                hidden_representations,
            ) = utils.get_predictions(
                self.model,
                batch,
                num_mc_dropout=self.num_mc_epistemic,
                aleatoric_model=self.aleatoric_model,
                evidential_model=self.evidential_model,
                num_mc_aleatoric=self.num_mc_aleatoric,
                ensemble_model=False,
                device=self.device,
            )
            mean_epistemic_unc_information = np.mean(epistemic_unc_predictions, axis=(1, 2))
            for i, idx in enumerate(batch["index"]):
                bald_objectives[unlabeled_indices == idx.item()] = mean_epistemic_unc_information[i]

        return unlabeled_indices[np.argpartition(bald_objectives, -sample_size)[-sample_size:]]
