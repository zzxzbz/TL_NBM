"""Gumbel-Softmax based feature selector.

Uses Gumbel-Softmax technique to select the most important subset of input features.
Can be used for feature selection and model interpretability.
"""

import pytorch_lightning as pl
import torch.nn as nn
import torch
import numpy as np


class GumbelSoftmaxFeatureSelector(pl.LightningModule):
    """Feature selector based on Gumbel-Softmax.
    
    This module uses Gumbel-Softmax technique to select the most important
    subset from input features.
    """
    
    def __init__(self, d: int, k: int, tau: float):
        """Initialize feature selector.
        
        Args:
            d: Input feature dimension
            k: Number of features to select
            tau: Gumbel-Softmax temperature parameter, controls sampling discreteness
        """
        super().__init__()
        self.d = d
        self.k = k
        self.tau = tau
        self.logits = Logits(d, d)
        self.gumbel_softmax_sampler = GumbelSoftmaxSampler(d, k, tau)
    
    def forward(self, x, condition=None, hard=False):
        """Perform feature selection.
        
        Args:
            x: Input features of shape (batch_size, d)
            condition: Optional condition vector for conditional feature selection
            hard: Whether to use hard (discrete) sampling
            
        Returns:
            Feature mask of shape (batch_size, d) with k positions set to 1, rest 0
        """
        batch_size = x.size(0)

        # Generate logits
        if condition is not None:
            logits = self.logits(condition)
        else:
            # Use random input for unconditional selection
            unconditional_input = torch.randn_like(x)
            logits = self.logits(unconditional_input)

        # Perform Gumbel-Softmax sampling
        samples = self.gumbel_softmax_sampler(logits, hard)

        # Check and handle NaN results
        if torch.isnan(samples).any():
            # Provide fallback sampling
            samples = self._generate_fallback_samples(logits, batch_size)
            
        return samples
    
    def _generate_fallback_samples(self, logits, batch_size):
        """Generate fallback samples when NaN occurs.
        
        Args:
            logits: Original logits
            batch_size: Batch size
            
        Returns:
            Fallback feature mask
        """
        print("Warning: samples contain NaN values, using fallback")
        samples = torch.zeros_like(logits)
        
        # Select top 50% features based on logits values
        input_feat_num = logits.size(1)
        topk_indices = logits.topk(int(input_feat_num * 0.5), dim=1)[1]
        
        for i in range(batch_size):
            samples[i, topk_indices[i]] = 1.0
            
        return samples


class GumbelSoftmaxSampler(pl.LightningModule):
    """Gumbel-Softmax sampler.
    
    Implements the Gumbel-Softmax reparameterization trick,
    enabling differentiable sampling from discrete distributions.
    """
    
    def __init__(self, d, k, tau):
        """Initialize sampler.
        
        Args:
            d: Feature dimension
            k: Number of features to select
            tau: Temperature parameter; smaller values produce more discrete samples
        """
        super().__init__()
        self.d = d
        self.k = k
        self.tau = tau

    def forward(self, logits, hard=False):
        """Perform Gumbel-Softmax sampling.
        
        Args:
            logits: Unnormalized log probabilities of shape (batch_size, d)
            hard: Whether to return discrete one-hot samples
            
        Returns:
            Sampling mask of shape (batch_size, d)
        """
        # Expand logits for k independent sampling
        logits = logits.unsqueeze(1).expand(-1, self.k, -1)
        
        # Apply Gumbel-Softmax
        samples = nn.functional.gumbel_softmax(logits, tau=self.tau, hard=hard)
        
        # Take max over k samples for the final mask
        samples = samples.max(dim=1)[0]

        return samples


class Logits(pl.LightningModule):
    """Logits generation network.
    
    Used to compute importance scores (logits) for each input feature.
    Includes batch normalization for training stability.
    """
    
    def __init__(self, d, hidden_dim):
        """Initialize Logits network.
        
        Args:
            d: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.d = d
        self.hidden_dim = hidden_dim

        # Network layers
        self.fc1 = nn.Linear(self.d, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(self.hidden_dim, self.d)

    def forward(self, x):
        """Compute feature importance scores.
        
        Args:
            x: Input features of shape (batch_size, d)
            
        Returns:
            Logits of shape (batch_size, d) representing each feature's importance
        """
        # Layer 1: Linear -> BatchNorm -> ReLU
        out1 = self.fc1(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        
        # Layer 2: Linear -> BatchNorm -> ReLU
        out2 = self.fc2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu2(out2)
        
        # Output layer: Linear (no BatchNorm, preserve logits distribution)
        out3 = self.fc3(out2)
        
        return out3


if __name__ == '__main__':
    # Test code
    selector = GumbelSoftmaxFeatureSelector(10, 5, 0.1)
    x = torch.randn(64, 10)
    y = selector(x)
    print("Selection result statistics:")
    print(f"Min: {y.min().item():.4f}, Max: {y.max().item():.4f}")
    print(f"Average non-zero elements: {(y > 0.5).float().sum(dim=1).mean().item():.2f}")
