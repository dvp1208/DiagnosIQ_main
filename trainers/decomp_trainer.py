from itertools import chain
import torch
import torch.nn.functional as F
import torch.optim as optim

from functions import SoftDiceLoss
from functions import FocalLoss
from functions import DiceCoefficient
from functions import OneHotEncoder
from functions import VGGLoss
from networks import Encoder
from networks import NormalEncoder
from networks import Decoder
from .decomp_base import DecompTrainerBase


class DecompTrainer(DecompTrainerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.config.losses.d_aac_margin_loss_type in {'norm', 'kldiv', 'none'}

    def configure_models(self):
        config = self.config.models

        if config.use_vae_nEnc:
            self.nEncoder = NormalEncoder(input_dim=config.input_dim,
                                          emb_dim=config.emb_dim,
                                          filters=config.enc_filters)
        else:
            self.nEncoder = Encoder(input_dim=config.input_dim,
                                    emb_dim=config.emb_dim,
                                    filters=config.enc_filters)

        if config.use_vae_aEnc:
            self.aEncoder = NormalEncoder(input_dim=config.input_dim,
                                          emb_dim=config.emb_dim,
                                          filters=config.enc_filters)
        else:
            self.aEncoder = Encoder(input_dim=config.input_dim,
                                    emb_dim=config.emb_dim,
                                    filters=config.enc_filters)

        self.iDecoder = Decoder(output_dim=config.input_dim,
                                emb_dim=config.emb_dim,
                                filters=config.dec_filters)

        self.lDecoder = Decoder(output_dim=config.seg_output_dim,
                                emb_dim=config.emb_dim,
                                filters=config.dec_filters)

        self.lEncoder = Encoder(input_dim=1,
                                emb_dim=config.emb_dim,
                                filters=config.enc_filters)

    def configure_losses(self):
        self.l_dice = SoftDiceLoss(ignore_index=self.config.losses.dice_loss.ignore_index)

        self.l_focal = FocalLoss(gamma=self.config.losses.focal_loss.gamma,
                                 alpha=self.config.losses.focal_loss.alpha)

        self.one_hot_encoder = OneHotEncoder(n_classes=self.config.metric.n_classes).forward

        self.dice_metric = DiceCoefficient(n_classes=self.config.metric.n_classes,
                                           index_to_class_name=self.index_to_class_name)

        self.l_perceptual = None

        if self.config.losses.use_perceptual:
            self.l_perceptual = VGGLoss()

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                               chain(self.nEncoder.parameters(),
                                     self.aEncoder.parameters(),
                                     self.iDecoder.parameters(),
                                     self.lDecoder.parameters(),
                                     self.lEncoder.parameters())),
                               self.config.optimizer.g_lr, [0.9, 0.9999], eps=1e-7,
                               weight_decay=self.config.optimizer.weight_decay)

        return optimizer

    def training_step(self, batches, batch_idx):
        n_batch, a_batch = batches

        n_rec_loss = 0.0
        e_rec_loss = 0.0
        seg_loss = 0.0
        consistent_loss = 0.0

        w = self.config.loss_weight
        use_vae_nEnc = self.config.models.use_vae_nEnc
        use_vae_aEnc = self.config.models.use_vae_aEnc
        detach_at_e_rec = self.config.losses.detach_at_e_rec
        minimize_aac_norm = self.config.losses.minimize_aac_norm
        d_aac_margin_loss_type = self.config.losses.d_aac_margin_loss_type

        for batch in [
            {'class_label': 'normal', 'batch': n_batch},
            {'class_label': 'abnormal', 'batch': a_batch},
        ]:

            image = batch['batch']['image']
            s_label = batch['batch']['seg_label']

            if use_vae_nEnc:
                nac, nac_mu, nac_logvar = self.nEncoder(image)
            else:
                nac = self.nEncoder(image)

            if use_vae_aEnc:
                aac, aac_mu, aac_logvar = self.aEncoder(image)
            else:
                aac = self.aEncoder(image)

            if batch['class_label'] == 'normal':
                if 'nac' in detach_at_e_rec and 'aac' in detach_at_e_rec:
                    e_rec = self.iDecoder(nac.detach() + aac.detach())

                elif 'nac' in detach_at_e_rec:
                    e_rec = self.iDecoder(nac.detach() + aac)

                elif 'aac' in detach_at_e_rec:
                    e_rec = self.iDecoder(nac + aac.detach())

                else:
                    e_rec = self.iDecoder(nac + aac)

                n_rec = self.iDecoder(nac)

            elif batch['class_label'] == 'abnormal':
                _nac = nac_mu if use_vae_nEnc else nac

                if 'nac' in detach_at_e_rec and 'aac' in detach_at_e_rec:
                    e_rec = self.iDecoder(_nac.detach() + aac.detach())

                elif 'nac' in detach_at_e_rec:
                    e_rec = self.iDecoder(_nac.detach() + aac)

                elif 'aac' in detach_at_e_rec:
                    e_rec = self.iDecoder(_nac + aac.detach())

                else:
                    e_rec = self.iDecoder(_nac + aac)

                n_rec = None

            a_seg = self.lDecoder(aac)
            r_aac = self.lEncoder(s_label.unsqueeze(1).float())

            # calculation of loss functions.
            if batch['class_label'] == 'normal':
                n_rec_loss += w.recon * self.l_recon(n_rec, image, mask=None, mode='none')
                e_rec_loss += w.recon * self.l_recon(e_rec, image, mask=None, mode='none')

            elif batch['class_label'] == 'abnormal':
                n_rec_loss += 0.0

                if self.config.losses.use_emphasis:
                    a_mask = (s_label > 0).unsqueeze(1).expand_as(image)
                    e_rec_loss += w.recon * self.l_recon(e_rec, image, mask=a_mask, mode='emphasize')

                else:
                    e_rec_loss += w.recon * self.l_recon(e_rec, image, mask=None, mode='none')

            seg_loss += w.seg * self.l_seg(a_seg, s_label)

            consistent_loss += w.consistent * self.l_consistent(aac, r_aac)

            if batch['class_label'] == 'normal':
                if use_vae_nEnc:
                    h_nac_vae = self.l_reg(nac_mu, nac_logvar)
                else:
                    h_nac_vae = 0.0

                h_nac_vae_loss = w.h_nac_vae * h_nac_vae

                if use_vae_aEnc:
                    h_aac_vae = self.l_reg(aac_mu, aac_logvar)
                else:
                    h_aac_vae = 0.0

                h_aac_vae_loss = w.h_aac_vae * h_aac_vae

                if minimize_aac_norm:
                    h_aac_norm = self.l_norm(aac_mu if use_vae_aEnc else aac)
                else:
                    h_aac_norm = 0.0

                h_aac_norm_loss = w.h_aac_norm * h_aac_norm

                h_nac = nac

            elif batch['class_label'] == 'abnormal':
                d_nac_vae_loss = 0.0

                if use_vae_aEnc:
                    d_aac_vae = self.l_reg(aac_mu, aac_logvar)
                else:
                    d_aac_vae = 0.0

                d_aac_vae_loss = w.d_aac_vae * d_aac_vae

                _aac = aac_mu if use_vae_aEnc else aac

                if d_aac_margin_loss_type == 'norm':
                    d_aac_margin = self.l_norm(_aac)

                elif d_aac_margin_loss_type == 'kldiv':
                    d_aac_margin = self.l_kldiv(nac.detach() + _aac, target=h_nac.detach())

                elif d_aac_margin_loss_type == 'none':
                    d_aac_margin = 0.0

                d_aac_margin_loss = w.d_aac_margin * F.relu(w.margin - d_aac_margin)

        

        total_loss = n_rec_loss + e_rec_loss + seg_loss + consistent_loss \
                   + h_nac_vae_loss + h_aac_vae_loss + h_aac_norm_loss \
                   + d_aac_vae_loss + d_aac_margin_loss
        
        self.log('epoch', torch.tensor([self.current_epoch]).type_as(total_loss), sync_dist=True)
        self.log('iteration', torch.tensor([self.global_step]).type_as(total_loss), sync_dist=True)

        self.log('total_loss', total_loss, prog_bar=True)
        self.log('n_rec_loss', n_rec_loss, prog_bar=True)
        self.log('e_rec_loss', e_rec_loss, prog_bar=True)
        self.log('seg_loss', seg_loss, prog_bar=True)
        self.log('consistent_loss', consistent_loss, prog_bar=True)

        self.log('h_nac_vae_loss', h_nac_vae_loss, prog_bar=True)
        self.log('h_aac_vae_loss', h_aac_vae_loss, prog_bar=True)
        self.log('h_aac_norm_loss', h_aac_norm_loss, prog_bar=True)
        self.log('d_aac_vae_loss', d_aac_vae_loss, prog_bar=True)
        self.log('d_aac_margin_loss', d_aac_margin_loss, prog_bar=True)

        self.log('h_nac_vae', h_nac_vae, prog_bar=True)
        self.log('h_aac_vae', h_aac_vae, prog_bar=True)
        self.log('h_aac_norm', h_aac_norm, prog_bar=True)
        self.log('d_aac_vae', d_aac_vae, prog_bar=True)
        self.log('d_aac_margin', d_aac_margin, prog_bar=True)

        return total_loss
