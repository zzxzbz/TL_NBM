"""
NBM residual transfer learning (FT).

- Backbone: cross-well pretrained SparseNBMROPModel, parameters frozen, inference uses hard Gumbel.
- Residual net: fits y - f_base(x), final prediction y_hat = f_base(x) + f_res(x).
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.ft_residual_model import _ResidualNet
from models.sparsenbm import SparseNBMROPModel

torch.set_default_dtype(torch.float32)


class NBMFTResidualModel(pl.LightningModule):
    """NBM backbone + MLP residual, only residual network is trained."""

    def __init__(
        self,
        backbone_model=None,
        params=None,
        backbone_checkpoint_path=None,
        input_data_range=None,
    ):
        super().__init__()

        if params is None:
            raise ValueError("params cannot be None")

        if backbone_checkpoint_path is not None:
            params['backbone_checkpoint_path'] = backbone_checkpoint_path
        if input_data_range is not None:
            params['input_data_range'] = input_data_range

        self.save_hyperparameters(params)
        self.params = params

        if backbone_model is not None:
            self.backbone = backbone_model
        elif self.hparams.get("backbone_checkpoint_path"):
            self.backbone = SparseNBMROPModel.load_from_checkpoint(
                self.hparams["backbone_checkpoint_path"],
                input_data_range=self.hparams.get("input_data_range"),
            )
        else:
            raise ValueError("Must provide backbone_model or backbone_checkpoint_path")

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.residual_net = _ResidualNet(self.hparams)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def _backbone_forward(self, x):
        self.backbone.eval()
        pred = self.backbone.forward(x, hard=True)
        return pred.squeeze(-1) if pred.dim() > 1 else pred

    def _flatten_target(self, y):
        return y.squeeze(-1) if y.dim() > 1 else y

    def forward(self, batch):
        x = batch['input']
        base = self._backbone_forward(x)
        delta = self.residual_net(x).squeeze(-1)
        return base + delta

    def training_step(self, batch, batch_idx):
        x = batch['input']
        y = self._flatten_target(batch['target'])

        base = self._backbone_forward(x)
        residual_label = y - base

        delta = self.residual_net(x).squeeze(-1)
        loss = nn.functional.mse_loss(delta, residual_label)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['input']
        y = self._flatten_target(batch['target'])

        base = self._backbone_forward(x)
        delta = self.residual_net(x).squeeze(-1)

        residual_label = y - base
        residual_loss = nn.functional.mse_loss(delta, residual_label)
        final_loss = nn.functional.mse_loss(base + delta, y)

        self.log('val_loss', residual_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_final_mse', final_loss, prog_bar=True, on_step=False, on_epoch=True)
        return residual_loss

    def test_step(self, batch, batch_idx):
        x = batch['input']
        y = self._flatten_target(batch['target'])

        base = self._backbone_forward(x)
        delta = self.residual_net(x).squeeze(-1)

        loss = nn.functional.mse_loss(base + delta, y)
        self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.residual_net.parameters(),
            lr=self.params['learning_rate'],
            weight_decay=self.params.get('weight_decay', 0.0),
        )

    @torch.no_grad()
    def predict(self, batch):
        x = batch['input'].to(self.device)
        base = self._backbone_forward(x)
        delta = self.residual_net(x).squeeze(-1)
        return base + delta
