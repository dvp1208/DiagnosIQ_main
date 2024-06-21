import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.distributed as dist
from pytorch_lightning.trainer.supporters import CombinedLoader
from kornia.morphology import dilation
from kornia.morphology import erosion

import functions.pytorch_ssim as pytorch_ssim
from dataio import get_data_loader
from utils import minmax_norm
from utils import to_cpu


MORPHO_KERNEL = 15


def getattr_else_none(config, attr):
    if hasattr(config, attr):
        return getattr(config, attr)
    else:
        return None


class DecompTrainerBase(pl.LightningModule):

    def __init__(self, config, save_dir_path, automatic_optimization, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.save_dir_path = save_dir_path
        self.automatic_optimization = automatic_optimization

        self.class_name_to_index = self.config.metric.class_name_to_index._asdict()
        self.index_to_class_name = {v: k for k, v in self.class_name_to_index.items()}

        self.configure_models()
        self.configure_optimizers()
        self.configure_losses()

    ######################################################
    #
    # DataLoaders
    #
    ######################################################

    def train_dataloader(self):
        if self.config.run.training_dataset == 'concat_dataset':
            return [
                get_data_loader(
                    dataset_name=self.config.dataset.dataset_name,
                    modalities=self.config.dataset.modalities,
                    root_dir_paths=self.config.dataset.train_root_dir_paths,
                    use_augmentation=True,
                    use_shuffle=True,
                    batch_size=self.config.dataset.batch_size // 2,
                    num_workers=self.config.dataset.num_workers // 2,
                    drop_last=True,
                    initial_randomize=True,
                    patient_ids=None,
                    dataset_class='normal',
                    window_width=getattr_else_none(self.config.dataset, 'window_width'),
                    window_center=getattr_else_none(self.config.dataset, 'window_center'),
                    window_scale=getattr_else_none(self.config.dataset, 'window_scale'),
                ),
                get_data_loader(
                    dataset_name=self.config.dataset.dataset_name,
                    modalities=self.config.dataset.modalities,
                    root_dir_paths=self.config.dataset.train_root_dir_paths,
                    use_augmentation=True,
                    use_shuffle=True,
                    batch_size=self.config.dataset.batch_size // 2,
                    num_workers=self.config.dataset.num_workers // 2,
                    drop_last=True,
                    initial_randomize=True,
                    patient_ids=None,
                    dataset_class='abnormal',
                    window_width=getattr_else_none(self.config.dataset, 'window_width'),
                    window_center=getattr_else_none(self.config.dataset, 'window_center'),
                    window_scale=getattr_else_none(self.config.dataset, 'window_scale'),
                )
            ]

        elif self.config.run.training_dataset == 'normal_dataset':
            return get_data_loader(
                dataset_name=self.config.dataset.dataset_name,
                modalities=self.config.dataset.modalities,
                root_dir_paths=self.config.dataset.train_root_dir_paths,
                use_augmentation=True,
                use_shuffle=True,
                batch_size=self.config.dataset.batch_size // 2,
                num_workers=self.config.dataset.num_workers // 2,
                drop_last=True,
                initial_randomize=True,
                patient_ids=None,
                dataset_class='normal',
                window_width=getattr_else_none(self.config.dataset, 'window_width'),
                window_center=getattr_else_none(self.config.dataset, 'window_center'),
                window_scale=getattr_else_none(self.config.dataset, 'window_scale'),
            )

        elif self.config.run.training_dataset == 'abnormal_dataset':
            return get_data_loader(
                dataset_name=self.config.dataset.dataset_name,
                modalities=self.config.dataset.modalities,
                root_dir_paths=self.config.dataset.train_root_dir_paths,
                use_augmentation=True,
                use_shuffle=True,
                batch_size=self.config.dataset.batch_size // 2,
                num_workers=self.config.dataset.num_workers // 2,
                drop_last=True,
                initial_randomize=True,
                patient_ids=None,
                dataset_class='abnormal',
                window_width=getattr_else_none(self.config.dataset, 'window_width'),
                window_center=getattr_else_none(self.config.dataset, 'window_center'),
                window_scale=getattr_else_none(self.config.dataset, 'window_scale'),
            )

    def val_dataloader(self):
        return CombinedLoader({
            'normal': get_data_loader(
                dataset_name=self.config.dataset.dataset_name,
                modalities=self.config.dataset.modalities,
                root_dir_paths=self.config.dataset.train_root_dir_paths,
                use_augmentation=False,
                use_shuffle=False,
                batch_size=self.config.dataset.batch_size // 2,
                num_workers=self.config.dataset.num_workers // 2,
                drop_last=True,
                initial_randomize=True,
                patient_ids=None,
                dataset_class='normal',
                window_width=getattr_else_none(self.config.dataset, 'window_width'),
                window_center=getattr_else_none(self.config.dataset, 'window_center'),
                window_scale=getattr_else_none(self.config.dataset, 'window_scale'),
            ),
            'abnormal': get_data_loader(
                dataset_name=self.config.dataset.dataset_name,
                modalities=self.config.dataset.modalities,
                root_dir_paths=self.config.dataset.train_root_dir_paths,
                use_augmentation=False,
                use_shuffle=False,
                batch_size=self.config.dataset.batch_size // 2,
                num_workers=self.config.dataset.num_workers // 2,
                drop_last=True,
                initial_randomize=True,
                patient_ids=None,
                dataset_class='abnormal',
                window_width=getattr_else_none(self.config.dataset, 'window_width'),
                window_center=getattr_else_none(self.config.dataset, 'window_center'),
                window_scale=getattr_else_none(self.config.dataset, 'window_scale'),
            )
        }, 'min_size')

    def test_dataloader(self):
        return get_data_loader(
            dataset_name=self.config.dataset.dataset_name,
            modalities=self.config.dataset.modalities,
            root_dir_paths=self.config.dataset.test_root_dir_paths,
            use_augmentation=False,
            use_shuffle=False,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            drop_last=False,
            initial_randomize=False,
            patient_ids=self.config.dataset.patient_ids,
            window_width=getattr_else_none(self.config.dataset, 'window_width'),
            window_center=getattr_else_none(self.config.dataset, 'window_center'),
            window_scale=getattr_else_none(self.config.dataset, 'window_scale'),
        )

    ######################################################
    #
    # Loss functions
    #
    ######################################################

    def l_recon(self, recon, target, mask=None, mode='none'):
        if self.config.losses.use_perceptual:
            return self._l_perceptual_recon(recon, target, mask, mode)
        else:
            return self._l_recon(recon, target, mask, mode)

    def _l_perceptual_recon(self, recon, target, mask=None, mode='none'):
        assert mode in ['none', 'crop', 'emphasize']

        if mode == 'none':
            loss = F.mse_loss(recon, target, reduction='mean') \
                 + self.l_perceptual(recon, target, mask=None)

        elif mode == 'crop':
            loss = F.mse_loss(recon[mask], target[mask], reduction='mean')
            kernel = torch.ones(MORPHO_KERNEL, MORPHO_KERNEL).type_as(recon)
            vgg_mask = erosion(mask.float(), kernel)
            loss += self.l_perceptual(recon, target, mask=vgg_mask)

        elif mode == 'emphasize':
            loss = F.mse_loss(recon, target, reduction='mean') \
                 + self.l_perceptual(recon, target, mask=None)

            if mask.byte().any().item() == 1:
                loss += F.mse_loss(recon[mask], target[mask], reduction='mean')
                kernel = torch.ones(MORPHO_KERNEL, MORPHO_KERNEL).type_as(recon)
                vgg_mask = dilation(mask.float(), kernel)
                loss += self.l_perceptual(recon, target, mask=vgg_mask)

        return loss

    def _l_recon(self, recon, target, mask=None, mode='none'):
        assert mode in ['none', 'crop', 'emphasize']
        ssim_loss = pytorch_ssim.SSIM(window_size=11)

        if mode == 'none':
            loss = F.mse_loss(recon, target, reduction='sum') \
                 + (1.0 - ssim_loss(recon, target)) * torch.numel(recon)

        elif mode == 'crop':
            loss = F.mse_loss(recon[mask], target[mask], reduction='sum') \
                 + (1.0 - ssim_loss(recon, target, mask)) * torch.numel(recon[mask])

        elif mode == 'emphasize':
            loss = F.mse_loss(recon, target, reduction='sum') \
                 + (1.0 - ssim_loss(recon, target)) * torch.numel(recon)

            if mask.byte().any().item() == 1:
                loss += F.mse_loss(recon[mask], target[mask], reduction='sum') \
                     + (1.0 - ssim_loss(recon, target, mask)) * torch.numel(recon[mask])

        return loss / recon.numel()

    def l_seg(self, seg_logit, seg_label):
        target = self.one_hot_encoder(seg_label)
        dice_loss = self.l_dice(seg_logit, target)
        focal_loss = self.l_focal(seg_logit, target)
        return dice_loss + focal_loss

    def l_consistent(self, aac, r_aac):
        return (
            torch.abs(aac.detach() - r_aac) + torch.abs(aac - r_aac.detach())
        ).mean()

    def l_l1(self, code, target=None):
        if target is None:
            return torch.abs(code).mean()

        return F.l1_loss(code, target, reduction='mean')

    def l_l2(self, code, target=None):
        if target is None:
            return (torch.abs(code) ** 2).mean()

        return F.mse_loss(code, target, reduction='mean')

    def l_norm(self, code):
        return torch.norm(code)

    def l_reg(self, mu, log_var):
        loss = torch.mean(- 0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=1))
        return loss

    def l_kldiv(self, code, target=None):
        if target is None:
            target = torch.randn_like(code)

        return nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(code, dim=1), F.softmax(target, dim=1)
            )

    def calc_adaptive_weight(self, g_loss, d_loss, n_layer, a_layer, eps=1e-4, vmax=1e4):
        n_g_grads = torch.norm(torch.autograd.grad(g_loss, n_layer, retain_graph=True)[0])
        n_d_grads = torch.norm(torch.autograd.grad(d_loss, n_layer, retain_graph=True)[0])

        a_g_grads = torch.norm(torch.autograd.grad(g_loss, a_layer, retain_graph=True)[0])
        a_d_grads = torch.norm(torch.autograd.grad(d_loss, a_layer, retain_graph=True)[0])

        g_grads = n_g_grads + a_g_grads
        d_grads = n_d_grads + a_d_grads

        d_weight = g_grads / (d_grads + eps)
        d_weight = torch.clamp(d_weight, 0.0, vmax).detach()

        return d_weight

    ######################################################
    #
    # Validation step
    #
    ######################################################

    def inference(self, image, label):
        assert not self.training

        nac = self.nEncoder(image)
        aac = self.aEncoder(image)

        eac = nac + aac

        e_rec = self.iDecoder(eac)
        n_rec = self.iDecoder(nac)

        a_seg = self.lDecoder(aac)
        r_aac = self.lEncoder(label.unsqueeze(1).float())

        return {
            'nac': nac,
            'aac': aac,
            'eac': eac,
            'e_rec': e_rec,
            'n_rec': n_rec,
            'a_seg': a_seg,
            'r_aac': r_aac,
        }

    def concat_batches(self, batches):
        n_batch, a_batch = batches

        n_image = n_batch['image']
        n_label = n_batch['seg_label']
        n_class = n_batch['class_label']

        a_image = a_batch['image']
        a_label = a_batch['seg_label']
        a_class = a_batch['class_label']

        batch_size = n_image.size(0) + a_image.size(0)

        image = torch.cat([n_image, a_image], dim=0)
        s_label = torch.cat([n_label, a_label], dim=0)
        c_label = torch.cat([n_class, a_class], dim=0)

        rand_idx = torch.randperm(batch_size)

        image = image[rand_idx, ...]
        s_label = s_label[rand_idx, ...]
        c_label = c_label[rand_idx]

        return image, s_label, c_label

    def validation_step(self, batches, batch_idx):
        image, s_label, c_label = self.concat_batches(
            [batches['normal'], batches['abnormal']]
        )

        with torch.no_grad():
            out = self.inference(image, s_label)

        seg_logit = out['a_seg']
        seg_mask = seg_logit.argmax(1)
        dice = self.dice_metric(seg_logit, s_label)

        if batch_idx == 0:
            image = to_cpu(image)
            e_recon = to_cpu(out['e_rec'])
            n_recon = to_cpu(out['n_rec'])

            vmin = image.min()
            vmax = image.max()

            image = minmax_norm(image, vmin=vmin, vmax=vmax)
            e_recon = minmax_norm(e_recon, vmin=vmin, vmax=vmax)
            n_recon = minmax_norm(n_recon, vmin=vmin, vmax=vmax)

            s_label = to_cpu(s_label)
            s_output = to_cpu(seg_mask)

            n_images = min(self.config.save.n_save_images, image.size(0))

            if 'MICCAI' in self.config.dataset.dataset_name:
                save_modalities = ['t1ce']

                if 'flair' in self.config.dataset.modalities:
                    save_modalities.append('flair')

                for save_modality in save_modalities:
                    idx = self.config.dataset.modalities.index(save_modality)

                    save_img = image[:n_images, ...][:, idx, ...][:, np.newaxis, ...]
                    save_e_rec = e_recon[:n_images, ...][:, idx, ...][:, np.newaxis, ...]
                    save_n_rec = n_recon[:n_images, ...][:, idx, ...][:, np.newaxis, ...]

                    image_grid = torch.cat([save_img, save_e_rec, save_n_rec])

                    self.logger.log_images(
                        save_modality,
                        image_grid,
                        self.current_epoch,
                        self.global_step,
                        nrow=n_images,
                    )

            elif 'Lung' in self.config.dataset.dataset_name:
                idx = 1

                save_img = image[:n_images, ...][:, idx, ...][:, np.newaxis, ...]
                save_e_rec = e_recon[:n_images, ...][:, idx, ...][:, np.newaxis, ...]
                save_n_rec = n_recon[:n_images, ...][:, idx, ...][:, np.newaxis, ...]

                image_grid = torch.cat([save_img, save_e_rec, save_n_rec])

                self.logger.log_images(
                    'image',
                    image_grid,
                    self.current_epoch,
                    self.global_step,
                    nrow=n_images,
                )

            s_label = s_label[:n_images, ...].float()[:, np.newaxis, ...]
            s_output = s_output[:n_images, ...].float()[:, np.newaxis, ...]

            max_label_val = self.config.metric.n_classes - 1
            s_label /= max_label_val
            s_output /= max_label_val

            label_grid = torch.cat([s_label, s_output])

            self.logger.log_images(
                'labels',
                label_grid,
                self.current_epoch,
                self.global_step,
                nrow=n_images,
            )

        return dice

    def validation_epoch_end(self, outputs):
        metrics = {
            'epoch': self.current_epoch,
            'iteration': self.global_step,
        }

        for key in self.class_name_to_index.keys():
            avg_value = torch.stack([x[key] for x in outputs], dim=0)
            try:
                avg_value = self.all_gather(avg_value).mean()
            except AttributeError:
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
        image = batch['image']
        s_label = batch['seg_label']
        patient_ids = batch['patient_id']
        n_slices = batch['n_slice']

        self.test_save_dir_path = os.path.join(
            self.config.save.save_dir_path,
            self.config.save.test_save_dir_name,
            self.config.save.study_name,
        )

        with torch.no_grad():
            out = self.inference(image, s_label)

        nac = out['nac'].cpu().numpy()
        aac = out['aac'].cpu().numpy()
        eac = out['eac'].cpu().numpy()

        for i, patient_id in enumerate(patient_ids):
            n_slice = int(n_slices[i].item())

            _nac = nac[i, ...]
            _aac = aac[i, ...]
            _eac = eac[i, ...]

            save_dir_path = os.path.join(
                self.test_save_dir_path,
                patient_id,
            )

            os.makedirs(save_dir_path, exist_ok=True)

            np.save(os.path.join(
                save_dir_path, 'nac_' + str(n_slice).zfill(4) + '.npy'
            ), _nac)

            np.save(os.path.join(
                save_dir_path, 'aac_' + str(n_slice).zfill(4) + '.npy'
            ), _aac)

            np.save(os.path.join(
                save_dir_path, 'eac_' + str(n_slice).zfill(4) + '.npy'
            ), _eac)

    def test_epoch_end(self, outputs):
        return self.global_rank
