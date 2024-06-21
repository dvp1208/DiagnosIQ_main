import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import ConfusionMatrix

from dataio import get_data_loader
from utils import to_cpu
from networks import ResNet
from .decomp_base import getattr_else_none


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_distributed() -> bool:
    return get_world_size() > 1


class ResNetTrainer(pl.LightningModule):

    def __init__(self, config, save_dir_path, monitoring_metrics, automatic_optimization=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.save_dir_path = save_dir_path
        self.monitoring_metrics = monitoring_metrics
        self.automatic_optimization = automatic_optimization
        self.output_features_at_test = self.config.run.output_features_at_test

        self.resnet = ResNet(
            model_name=self.config.model.model_name,
            fine_tuning=self.config.model.fine_tuning,
            output_dim=1,
            extracted_layers=['avgpool'],
        )

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.config.loss_weight.pos_weight]))
        self.metric = ConfusionMatrix(num_classes=2, compute_on_step=True)

    def configure_optimizers(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.resnet.parameters()),
                          self.config.optimizer.lr, [0.9, 0.9999],
                          weight_decay=self.config.optimizer.weight_decay)

    ######################################################
    #
    # DataLoaders
    #
    ######################################################

    def train_dataloader(self):
        assert self.config.dataset.dataset_name in {'MICCAIBraTSDataset', 'LungDataset'}

        return get_data_loader(
            dataset_name=self.config.dataset.dataset_name,
            modalities=self.config.dataset.modalities,
            root_dir_paths=self.config.dataset.train_root_dir_paths,
            use_augmentation=True,
            use_shuffle=True,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            drop_last=False,
            initial_randomize=False,
            patient_ids=self.config.dataset.patient_ids,
            dataset_class=None,
            window_width=getattr_else_none(self.config.dataset, 'window_width'),
            window_center=getattr_else_none(self.config.dataset, 'window_center'),
            window_scale=getattr_else_none(self.config.dataset, 'window_scale'),
        )

    def val_dataloader(self):
        assert self.config.dataset.dataset_name in {'MICCAIBraTSDataset', 'LungDataset'}

        return get_data_loader(
            dataset_name=self.config.dataset.dataset_name,
            modalities=self.config.dataset.modalities,
            root_dir_paths=self.config.dataset.train_root_dir_paths,
            use_augmentation=False,
            use_shuffle=False,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            drop_last=False,
            initial_randomize=True,
            patient_ids=self.config.dataset.patient_ids,
            dataset_class=None,
            window_width=getattr_else_none(self.config.dataset, 'window_width'),
            window_center=getattr_else_none(self.config.dataset, 'window_center'),
            window_scale=getattr_else_none(self.config.dataset, 'window_scale'),
        )

    def test_dataloader(self):
        assert self.config.dataset.dataset_name in {'MICCAIBraTSDataset', 'LungDataset'}

        return get_data_loader(
            dataset_name=self.config.dataset.dataset_name,
            modalities=self.config.dataset.modalities,
            root_dir_paths=self.config.dataset.test_root_dir_paths,
            use_augmentation=False,
            use_shuffle=False,
            batch_size=self.config.dataset.test_batch_size,
            num_workers=self.config.dataset.num_workers,
            drop_last=False,
            initial_randomize=False,
            patient_ids=self.config.dataset.patient_ids,
            dataset_class=None,
            window_width=getattr_else_none(self.config.dataset, 'window_width'),
            window_center=getattr_else_none(self.config.dataset, 'window_center'),
            window_scale=getattr_else_none(self.config.dataset, 'window_scale'),
        )

    ######################################################
    #
    # Training step
    #
    ######################################################

    def training_step(self, batch, batch_idx):
        image = batch['image']
        c_label = batch['class_label'].float()

        out, features = self.resnet(image)

        total_loss = self.criterion(out, c_label)

        self.log('epoch', self.current_epoch)
        self.log('iteration', self.global_step)
        self.log('total_loss', total_loss, prog_bar=True)

        return total_loss

    ######################################################
    #
    # Validation step
    #
    ######################################################

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        c_label = batch['class_label'].float()

        with torch.no_grad():
            out, features = self.resnet(image)
            total_loss = self.criterion(out, c_label)
            conf = self.metric(out, c_label.long())

        tn, fp, fn, tp = conf.flatten()

        accuracy = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (fp + tn)

        return {
            'total_loss': total_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
        }

    def validation_epoch_end(self, outputs):
        metrics = {
            'epoch': self.current_epoch,
            'iteration': self.global_step,
        }

        for key in self.monitoring_metrics:
            if key in outputs[0].keys():
                avg_value = torch.stack([x[key] for x in outputs], dim=0)

                if is_distributed():
                    avg_value = self.all_gather(avg_value).mean()
                else:
                    avg_value = avg_value.mean()

                metrics.update({
                    key: avg_value.item(),
                })

        self.logger.log_val_metrics(metrics)

    ######################################################
    #
    # Testing step
    #
    ######################################################

    def test_step(self, batch, batch_idx):
        if self.output_features_at_test:
            self._output_features(batch, batch_idx)

        return self._summarize_accuracy(batch, batch_idx)

    def _summarize_accuracy(self, batch, batch_idx):
        image = batch['image']
        c_label = batch['class_label'].float()

        with torch.no_grad():
            out, features = self.resnet(image)
            total_loss = self.criterion(out, c_label)
            conf = self.metric(out, c_label.long())

        tn, fp, fn, tp = conf.flatten()

        accuracy = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (fp + tn)

        result = {
            'total_loss': total_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
        }

        return result

    def test_epoch_end(self, outputs):
        metrics = {
            'epoch': self.current_epoch,
            'iteration': self.global_step,
        }

        for key in self.monitoring_metrics:
            if key in outputs[0].keys():
                avg_value = torch.stack([x[key] for x in outputs if not torch.isnan(x[key])], dim=0)

                if is_distributed():
                    avg_value = self.all_gather(avg_value)

                metrics.update({
                    key + '_mean': avg_value.mean().item(),
                    key + '_std': avg_value.std().item(),
                })

        self.logger.log_test_metrics(metrics)

    def _output_features(self, batch, batch_idx):
        patient_ids = batch['patient_id']
        n_slice = batch['n_slice']
        image = batch['image']

        with torch.no_grad():
            _, features = self.resnet(image)

        assert len(features) == 1
        features = features[0]
        features = to_cpu(features).numpy()

        for i in range(len(patient_ids)):
            patient_id = patient_ids[i]
            slice_num = n_slice[i].item()
            feature = features[i, ...]

            save_path = os.path.join(self.config.save.database_path, patient_id)
            os.makedirs(save_path, exist_ok=True)

            np.save(os.path.join(save_path, 'feature_{}'.format(str(slice_num).zfill(4))), feature)
