"""
Simple direct-difference fine-tuning model.
The fine-tuning model directly learns the difference between the backbone prediction and the true value.
Final prediction = backbone prediction + fine-tuned difference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

torch.set_default_dtype(torch.float32)

class SimpleDirectDifferenceNetwork(nn.Module):
    """Simple direct-difference network - learns the difference between backbone prediction and ground truth."""
    
    def __init__(self, params):
        super(SimpleDirectDifferenceNetwork, self).__init__()
        self.params = params
        # Simple MLP that directly outputs the difference
        self.network = nn.Sequential(
            nn.Linear(self.params['input_dim'], self.params['hidden_dim']),
            nn.BatchNorm1d(self.params['hidden_dim']),
            nn.LeakyReLU(0.5),  # Use LeakyReLU to allow negative values
            nn.Dropout(self.params['dropout']),

            nn.Linear(self.params['hidden_dim'], self.params['hidden_dim']),
            nn.BatchNorm1d(self.params['hidden_dim']),
            nn.LeakyReLU(0.5),
            nn.Dropout(self.params['dropout']),
            
            nn.Linear(self.params['hidden_dim'], 1),  # Direct difference output, no activation
        )
    
    def forward(self, x):
        """Forward pass - directly outputs the difference, no activation function."""
        difference = self.network(x)
         
        return difference


class SimpleDirectDifferenceModel(pl.LightningModule):
    """Simple direct-difference fine-tuning model."""
    
    def __init__(self, backbone_model=None, params=None, backbone_checkpoint_path=None):
        super(SimpleDirectDifferenceModel, self).__init__()
        
        self.params = params
        if params is not None:
            if backbone_checkpoint_path is not None:
                params['backbone_checkpoint_path'] = backbone_checkpoint_path
            self.save_hyperparameters(params)

        # If backbone model provided directly, use it; otherwise reconstruct from checkpoint path
        if backbone_model is not None:
            self.backbone = backbone_model
        elif hasattr(self, 'hparams') and 'backbone_checkpoint_path' in self.hparams:
            from models.CLROPModel import CLROPModel
            self.backbone = CLROPModel.load_from_checkpoint(
                self.hparams.backbone_checkpoint_path
            ).ROPModel
        else:
            raise ValueError("Must provide backbone_model or backbone_checkpoint_path")
            
        self.difference_net = SimpleDirectDifferenceNetwork(self.hparams)
        
        # Freeze backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Keep backbone in eval mode to prevent BN/Dropout fluctuations during training
        self.backbone.eval()
        
        # Disable automatic optimization
        self.automatic_optimization = False 
        
    def forward(self, batch):
        """Forward pass."""
        
        that_row_data = batch['input']
        # Only train difference net; backbone forward pass without gradient tracking
        with torch.no_grad():
            backbone_prediction = self.backbone({'input': that_row_data})

        difference = self.difference_net(that_row_data)
        
        # Final prediction = backbone prediction + difference
        final_prediction = backbone_prediction + difference
        
        return  final_prediction.squeeze()

    def training_step(self, batch, batch_idx):
        # Get optimizer
        newbitROP_opt = self.optimizers()
        # Zero gradients
        newbitROP_opt.zero_grad()

        y = batch['target'].to(self.device).squeeze()
        # Ensure input is on the correct device
        batch['input'] = batch['input'].to(self.device)
        y_hat = self.forward(batch)
        loss = nn.MSELoss()(y_hat, y)

        self.manual_backward(loss)
        # Update parameters
        newbitROP_opt.step()

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['target'].to(self.device).squeeze()
        batch['input'] = batch['input'].to(self.device)
        y_hat = self.forward(batch)
        loss = nn.MSELoss()(y_hat, y)

        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        y = batch['target'].to(self.device).squeeze()
        batch['input'] = batch['input'].to(self.device)
        y_hat = self.forward(batch)
        loss = nn.MSELoss()(y_hat, y)

        self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Get optimizer."""

        optimizer = torch.optim.Adam(self.difference_net.parameters(),lr=self.params['learning_rate'])
    
        return optimizer
    
    @torch.no_grad()
    def predict(self, input):
        that_row_data = input['input'].to(self.device)

        ROP_hat = self.backbone(that_row_data)+self.difference_net(that_row_data)

        return ROP_hat

    def loss(self, y_hat, y):
        """Compute loss - MSE loss only."""
        regression_loss = nn.MSELoss()(y_hat, y)
        return regression_loss
