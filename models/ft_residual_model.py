"""
FT residual fine-tuning model.

Core logic:
    - Backbone (base/cross-well model) parameters are fully frozen and kept in eval mode.
    - The backbone only provides base predictions f_base(x) and does not participate in gradient backprop.
    - The fine-tuning network f_FT(x) directly fits the residual "ground truth - backbone prediction":
            label_residual = y - f_base(x)
        training loss is MSE(f_FT(x), label_residual).
    - At inference / validation / test time, the final prediction is y_hat = f_base(x) + f_FT(x).
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

torch.set_default_dtype(torch.float32)


class _ResidualNet(nn.Module):
    """Residual prediction network: directly outputs the difference from the ground truth, no non-negative activation."""

    def __init__(self, params):
        super().__init__()
        in_dim = params['input_dim']
        hid = params['hidden_dim']
        drop = params.get('dropout', 0.0)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.BatchNorm1d(hid),
            nn.LeakyReLU(0.5),
            nn.Dropout(drop),

            nn.Linear(hid, hid),
            nn.BatchNorm1d(hid),
            nn.LeakyReLU(0.5),
            nn.Dropout(drop),

            nn.Linear(hid, hid),
            nn.BatchNorm1d(hid),
            nn.LeakyReLU(0.5),
            nn.Dropout(drop),

            nn.Linear(hid, 1),
        )

    def forward(self, x):
        return self.net(x)


class FTResidualModel(pl.LightningModule):
    """Residual fine-tuning model: backbone frozen, only residual network trained."""

    def __init__(self, backbone_model=None, params=None, backbone_checkpoint_path=None):
        super().__init__()

        if params is None:
            raise ValueError("params cannot be None")

        if backbone_checkpoint_path is not None:
            params['backbone_checkpoint_path'] = backbone_checkpoint_path

        self.save_hyperparameters(params)
        self.params = params

        if backbone_model is not None:
            self.backbone = backbone_model
        elif self.hparams.get("backbone_checkpoint_path"):
            from models.CLROPModel import CLROPModel
            self.backbone = CLROPModel.load_from_checkpoint(
                self.hparams["backbone_checkpoint_path"]
            )
        else:
            from models.CLROPModel import CLROPModel
            backbone_params = dict(params)
            # Backbone MLP hidden_dim may differ from residual net (e.g., 32 vs 8)
            if params.get("backbone_hidden_dim") is not None:
                backbone_params["hidden_dim"] = params["backbone_hidden_dim"]
            self.backbone = CLROPModel(backbone_params)

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
        """Backbone prediction: always in eval + no_grad, outputs 1-D tensor matching y shape."""
        self.backbone.eval()
        pred = self.backbone.ROPModel(x)
        return pred.squeeze(-1)

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
