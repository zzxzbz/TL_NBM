import pytorch_lightning as pl
import torch.nn as nn
import torch

# Set default data type to float32 (required for Mac compatibility)
torch.set_default_dtype(torch.float32)

# Simple MLP model for ROP prediction
class CLROPModel(pl.LightningModule):

    def __init__(self, params, coefmodel = None):
        super(CLROPModel, self).__init__()

        self.params = params

        self.save_hyperparameters(params)

        # Disable automatic optimization since we handle it manually
        self.automatic_optimization = False 

        # Load pretrained model if provided
        if coefmodel is None:
            self.ROPModel= ROPModel(params)
        else:
            self.ROPModel = coefmodel
        
    def forward(self, batch):

        that_row_data = batch['input']

        ROP_hat = self.ROPModel(that_row_data)

        return ROP_hat.squeeze()

    def loss(self, y_hat, y):
        
        rop_loss = nn.MSELoss()(y_hat, y)
        
        return rop_loss


    def training_step(self, batch, batch_idx):

        # Get optimizer
        newbitROP_opt = self.optimizers()
        # Zero gradients
        newbitROP_opt.zero_grad()

        y = batch['target'].squeeze()
        y_hat = self.forward(batch) 
        loss = nn.MSELoss()(y_hat, y)

        self.manual_backward(loss)
        # Update parameters
        newbitROP_opt.step()

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        y = batch['target'].squeeze()

        y_hat = self.forward(batch)
        
        loss = nn.MSELoss()(y_hat, y)
        
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        y = batch['target'].squeeze()

        y_hat = self.forward(batch)
        
        loss = nn.MSELoss()(y_hat, y)
        
        self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.ROPModel.parameters(),
            lr=self.params['learning_rate'],
            weight_decay=self.params.get('weight_decay', 0.0),
        )
    
    @torch.no_grad()
    def predict(self, input):
        that_row_data = input['input'].to(self.device)
        ROP_hat = self.ROPModel(that_row_data)
        return ROP_hat


class ROPModel(nn.Module):
    """Simple 3-layer MLP for ROP prediction."""
    def __init__(self, params):
        super(ROPModel, self).__init__()

        self.params = params
        self.ROPModel = nn.Sequential(

            nn.Linear(self.params['input_dim'], self.params['hidden_dim']),
            nn.BatchNorm1d(self.params['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(self.params['dropout']),

            nn.Linear(self.params['hidden_dim'], self.params['hidden_dim']),
            nn.BatchNorm1d(self.params['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(self.params['dropout']),

            nn.Linear(self.params['hidden_dim'], self.params['hidden_dim']),
            nn.BatchNorm1d(self.params['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(self.params['dropout']),

            # Linear output layer (no activation, aligned with NBM)
            nn.Linear(self.params['hidden_dim'], 1),
        )
    
    def forward(self, that_row_data):
        return self.ROPModel(that_row_data)    
