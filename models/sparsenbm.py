import pytorch_lightning as pl
import torch.nn as nn
import torch
import os
import time
import numpy as np
from scipy.interpolate import griddata
from models.nbm import ConceptNBMNary
#from models.nbm_cfc import ConceptNBMNaryCfC as ConceptNBMNary
#from models.nbm_GRN import ConceptNBMNary
from models.gumbelsoftmaxselector import GumbelSoftmaxFeatureSelector
#from models.mlp import MLPROPModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


class SparseNBMROPModel(pl.LightningModule):
    """ROP prediction model using Gumbel-Softmax feature selector.
    
    This model uses Gumbel-Softmax mechanism to select the most important features,
    then performs ROP (Rate of Penetration) prediction based on the selected features.
    """
    
    def __init__(self, params, input_data_range = None):
        """Initialize model.
        
        Args:
            params: Dictionary containing model parameters, must include:
                - nbm_params: Parameters for ConceptNBMNary
                - gumbel_layer_params: Parameters for GumbelSoftmaxFeatureSelector
                - optimizer: Optimizer configuration
        """
        super().__init__()
        self.save_hyperparameters(params)
        self.params = params
        
        self.input_data_range = input_data_range
        input_data_min = torch.from_numpy(self.input_data_range[0]).to(self.device).float().reshape(1, -1)
        input_data_max = torch.from_numpy(self.input_data_range[1]).to(self.device).float().reshape(1, -1)

        if self.input_data_range is not None:
            self.input_data_range = (input_data_min, input_data_max)
        # Initialize main components
        # self.rop_model = MLPROPModel(**params['mlp_params'])
        self.rop_model = ConceptNBMNary(**params['nbm_params'])
        if params['is_sparse']:
            self.gumbel_selector = GumbelSoftmaxFeatureSelector(**params['gumbel_layer_params'])
        else:
            self.gumbel_selector = None

    def forward(self, x, known_mask=None, hard=False):
        """Model forward pass.
        
        Args:
            x: Input features of shape (batch_size, features)
            known_mask: Optional predefined feature mask
            hard: Whether to use hard (discrete) Gumbel-Softmax sampling
            
        Returns:
            Predicted ROP value
        """
        if known_mask is None:
            # Use Gumbel-Softmax to select features
            if self.gumbel_selector is not None:
                feature_mask = self.gumbel_selector(x, hard=hard)
            else:
                feature_mask = torch.ones_like(x)
            masked_input = x * feature_mask
            output = self.rop_model(masked_input, feature_mask)
        else:
            # Use provided mask
            masked_input = x * known_mask
            output = self.rop_model(masked_input, known_mask)
        
        return output.squeeze()

    def _loss(self, batch, stage):
        """Compute loss for given batch.
        
        Args:
            batch: Batch data containing 'input' and 'target'
            stage: Training stage ('train', 'val', 'test')
            
        Returns:
            Computed loss value
        """
        x = batch['input']
        y = batch['target']
        
        # Soft sampling during training, hard sampling during validation/test
        is_training = stage == 'train'
        y_hat = self.forward(x, hard=(not is_training))
        
        loss = nn.MSELoss()(y_hat, y.squeeze())
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._loss(batch, 'train')

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._loss(batch, 'val')
        return self._loss(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._loss(batch, 'test')
    
    @torch.no_grad()
    def predict(self, x, known_mask=None):
        """Gradient-free inference."""
        x = x.to(self.device).float()

        if known_mask is None:
            return self.forward(x, hard=True)
        else:
            masked_input = x * known_mask
            return self.rop_model(masked_input, known_mask)
        
    def obtain_feature_mask(self):
        """Get feature mask."""
        return self._determine_feature_mask(mask_samples=128)


    # def on_validation_epoch_end(self, epoch):
    #     """Callback at end of validation epoch.
        
    #     Generates simplified interpretability charts at end of validation for monitoring training progress.
    #     """
    #     try:
    #         # Create temp directory for validation plots
    #         temp_dir = os.path.join(self.logger.log_dir if self.logger else './logs', 'validation_plots')
            
    #         # Use fewer samples to speed up validation
    #         self.evaluation(
    #             save_dir=temp_dir,
    #             feat_name=None,  # No feature names needed during validation
    #             scaler=None,     # Use standardized data during validation
    #             num_samples=1024,  # Reduce number of samples
    #             mask_samples=64,   # Reduce mask sample count
    #             grid_resolution=50  # Lower grid resolution
    #         )
            
    #     except Exception as e:
    #         print(f"验证期间生成可解释性图表失败: {e}")
    #         # Validation failure should not interrupt training

    def test_step(self, batch, batch_idx):
        """测试步骤
        
        Args:
            batch: 测试批次
            batch_idx: 批次索引
            
        Returns:
            测试损失
        """
        return self._loss(batch, 'test')

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Get and convert learning rate
        lr = self.params["optimizer"].get("learning_rate", 1e-3)
        if isinstance(lr, str):
            lr = float(lr)
        
        # Get weight decay
        weight_decay = self.params["optimizer"].get("weight_decay", 1e-4)
        
        # Create Adam optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            eps=1e-8,
            weight_decay=weight_decay
        )
        
        # Optional learning rate scheduler
        if self.params.get("scheduler", {}).get("use_scheduler", False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5,  
                patience=5,
                min_lr=1e-6
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch"
                }
            }
        
        return optimizer

    @staticmethod
    def _snap_bit_diameter_normalized_tensor(x_z, scaler, feat_columns, physical_levels_np):
        """Replace BIT_DIAMETER column in normalized tensor with random z values corresponding to physical levels observed in data."""
        if physical_levels_np is None or len(physical_levels_np) == 0:
            return x_z
        if feat_columns is None or "BIT_DIAMETER" not in feat_columns:
            return x_z
        idx = list(feat_columns).index("BIT_DIAMETER")
        m = float(scaler.mean_[idx])
        s = float(scaler.scale_[idx])
        if s == 0.0:
            s = 1.0
        z_levels_np = (physical_levels_np.astype(np.float64) - m) / s
        device = x_z.device
        dtype = x_z.dtype
        z_levels = torch.as_tensor(z_levels_np, device=device, dtype=dtype)
        n = x_z.shape[0]
        pick = torch.randint(0, z_levels.shape[0], (n,), device=device)
        out = x_z.clone()
        out[:, idx] = z_levels[pick]
        return out

    @torch.no_grad()
    def explainable_prediction(self, x, save_dir=None, feat_name=None, scaler=None, 
                              num_samples=4096, mask_samples=128, grid_resolution=100,
                              bit_diameter_physical_levels=None, bit_diameter_feat_columns=None):
        """Explainable prediction for specific input."""
        if save_dir is None:
        if save_dir is None:
            raise ValueError("save_dir cannot be None, please specify a save directory")
            
        # Ensure input is on the correct device
        x = x.to(self.device).float()
        
        # Execute core explainability analysis pipeline
        viz_data = self._perform_explainability_analysis(
            target_input=x,
            feat_name=feat_name,
            scaler=scaler,
            num_samples=num_samples,
            mask_samples=mask_samples,
            bit_diameter_physical_levels=bit_diameter_physical_levels,
            bit_diameter_feat_columns=bit_diameter_feat_columns,
        )
        
        # Generate and save plots with target points
        self._create_and_save_plots(
            viz_data, save_dir, grid_resolution, 
            num_target_at_end=x.shape[0]
        )

        # Predict for target input

        return viz_data['final_output']

    @torch.no_grad()
    def explanation(self, save_dir=None, feat_name=None, scaler=None, 
                   num_samples=4096, mask_samples=128, grid_resolution=100,
                   bit_diameter_physical_levels=None, bit_diameter_feat_columns=None):
        """Global model explainability visualization.
        
        Generates interpretability charts showing impact of different feature combinations on outputs.
        """
        if save_dir is None:
        if save_dir is None:
            raise ValueError("save_dir cannot be None, please specify a save directory")
        
        if self.input_data_range is None:
            raise ValueError("Model input_data_range not set, cannot evaluate")
            
        try:
            # Execute core explainability analysis pipeline (without target input)
            viz_data = self._perform_explainability_analysis(
                target_input=None,
                feat_name=feat_name,
                scaler=scaler,
                num_samples=num_samples,
                mask_samples=mask_samples,
                bit_diameter_physical_levels=bit_diameter_physical_levels,
                bit_diameter_feat_columns=bit_diameter_feat_columns,
            )
            
            # Generate and save plots, without target points
            self._create_and_save_plots(viz_data, save_dir, grid_resolution)
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise RuntimeError(f"Model evaluation failed: {e}")
    
    def _perform_explainability_analysis(self, target_input=None, feat_name=None, 
                                       scaler=None, num_samples=4096, mask_samples=128,
                                       bit_diameter_physical_levels=None, bit_diameter_feat_columns=None):
        """Core explainability analysis pipeline.

        
        Args:
            target_input (torch.Tensor, optional): Target input data
            feat_name (list, optional): Feature name list
            scaler (sklearn.preprocessing, optional): Data scaler
            num_samples (int): Number of samples
            mask_samples (int): Mask sample count
            bit_diameter_physical_levels (np.ndarray, optional): BIT_DIAMETER discrete physical levels
            bit_diameter_feat_columns (list, optional): Column names matching scaler order
        
        Returns:
            dict: Visualization data
        """
        # Step 1: Determine feature mask
        feature_mask = self._determine_feature_mask(mask_samples)
        
        # Step 2: Generate evaluation data
        eval_data = self._generate_evaluation_data(
            num_samples,
            feature_mask,
            scaler,
            bit_diameter_physical_levels=bit_diameter_physical_levels,
            bit_diameter_feat_columns=bit_diameter_feat_columns,
        )
        
        # Step 3: Model inference
        model_outputs = self._perform_model_inference(eval_data)
        
        # Step 4: If target input exists, process target input predictions
        if target_input is not None:
            target_outputs = self._process_target_input(
                target_input, feature_mask, scaler
            )
            
            # Merge target input results into evaluation data
            eval_data = self._merge_target_data(eval_data, target_outputs['eval_data'])
            model_outputs = self._merge_target_data(model_outputs, target_outputs['model_outputs'])
        
        # Step 5: Prepare visualization data
        viz_data = self._prepare_visualization_data(
            eval_data, model_outputs, feat_name
        )
        if target_input is not None:
            viz_data['final_output'] = target_outputs['model_outputs']['out']
        
        return viz_data
    
    def _process_target_input(self, target_input, feature_mask, scaler):
        """Process target input prediction and data transform.
        
        Args:
            target_input (torch.Tensor): Target input data
            feature_mask (torch.Tensor): Feature mask
            scaler: Data scaler
            
        Returns:
            dict: Evaluation data and model outputs containing target input
        """
        # Predict for target input
        output, out_feats = self.rop_model(target_input, feature_mask, return_feats=True)
        
        # Inverse transform to obtain original-scale data
        if scaler is not None:
            try:
                x_inverse = target_input.clone().cpu().numpy().reshape(-1, scaler.n_features_in_)
                x_inverse = scaler.inverse_transform(x_inverse)
                x_inverse = torch.from_numpy(x_inverse).to(self.device).float()
            except Exception as e:
                print(f"Warning: target input data inverse transform failed: {e}")
                x_inverse = target_input.clone()
        else:
            x_inverse = target_input.clone()
        
        return {
            'eval_data': {
                'x_inverse': x_inverse
            },
            'model_outputs': {
                'out': output,
                'out_feats': out_feats
            }
        }
    
    def _merge_target_data(self, original_data, target_data):
        """Merge target data into original data.
        
        Args:
            original_data (dict): Original data dict
            target_data (dict): Target data dict
            
        Returns:
            dict: Merged data dict
        """
        merged_data = original_data.copy()
        
        for key, value in target_data.items():
            if key in merged_data:
                # If key exists, concatenate tensors
                merged_data[key] = torch.cat([merged_data[key], value], dim=0)
            else:
                # If key does not exist, add directly
                merged_data[key] = value
                
        return merged_data
    
    def _determine_feature_mask(self, mask_samples):
        """Determine feature selection mask.
        
        Args:
            mask_samples (int): Number of samples used to determine mask
            
        Returns:
            torch.Tensor: Feature mask of shape (1, num_features)
        """
        # Generate random samples to determine feature mask
        x_mask = self._generate_random_samples(mask_samples)
        x_mask = x_mask.to(self.device).float()
        
        # Use Gumbel selector to determine feature mask
        if self.gumbel_selector is not None:
            feature_mask = self.gumbel_selector(x_mask, hard=True)
        else:
            feature_mask = torch.ones_like(x_mask)
        
        # Average and binarize the mask
        feature_mask = (feature_mask.mean(0) > 0.5).float().reshape(1, -1)
        
        return feature_mask
    
    def _generate_random_samples(self, num_samples):
        """Generate specified number of random samples.
        
        Args:
            num_samples (int): Number of samples
            
        Returns:
            torch.Tensor: Random samples of shape (num_samples, num_features)
        """
        data_min, data_max = self.input_data_range
        data_min = data_min
        data_max = data_max
        return data_min + (data_max - data_min) * torch.rand(num_samples, data_min.shape[1])
    
    def _generate_evaluation_data(
        self,
        num_samples,
        feature_mask,
        scaler,
        bit_diameter_physical_levels=None,
        bit_diameter_feat_columns=None,
    ):
        """Generate evaluation data.
        
        input_data_range represents min/max of each feature in the **standardized (z) space**; background samples are model inputs z.
        Optionally snap the BIT_DIAMETER column to z values corresponding to observed physical levels, then inverse_transform
        to obtain physical-scale x_inverse for plotting.
        
        Args:
            num_samples (int): Number of samples
            feature_mask (torch.Tensor): Feature mask
            scaler: Data scaler
            bit_diameter_physical_levels: Discrete bit diameter physical values (1-D numpy)
            bit_diameter_feat_columns: Column names list matching scaler order
            
        Returns:
            dict: Contains z-space x, physical-scale x_inverse, masked_input
        """
        x_z = self._generate_random_samples(num_samples).to(self.device).float()
        if (
            scaler is not None
            and bit_diameter_physical_levels is not None
            and bit_diameter_feat_columns is not None
        ):
            x_z = self._snap_bit_diameter_normalized_tensor(
                x_z, scaler, bit_diameter_feat_columns, bit_diameter_physical_levels
            )

        if scaler is not None:
            try:
                x_inverse_phys = torch.from_numpy(
                    scaler.inverse_transform(x_z.detach().cpu().numpy())
                ).to(self.device).float()
            except Exception as e:
                print(f"Warning: data inverse transform failed: {e}")
                x_inverse_phys = x_z.clone()
        else:
            x_inverse_phys = x_z.clone()

        masked_input = x_z * feature_mask

        return {
            'x': x_z,
            'x_inverse': x_inverse_phys,
            'masked_input': masked_input,
            'feature_mask': feature_mask,
        }
    
    def _perform_model_inference(self, eval_data):
        """Perform model inference.
        
        Args:
            eval_data (dict): Evaluation data
            
        Returns:
            dict: Dictionary containing model output, features, and weights
        """
        try:
            # Model forward pass
            out, out_feats = self.rop_model(
                eval_data['masked_input'], 
                eval_data['feature_mask'], 
                return_feats=True
            )
            
            # Extract classifier weights and bias
            weights = self.rop_model.classifier.weight.cpu().numpy().reshape(-1)
            bias = self.rop_model.classifier.bias.cpu().numpy().reshape(-1)
            
            return {
                'out': out,
                'out_feats': out_feats,
                'weights': weights,
                'bias': bias
            }
            
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {e}")
    
    def _prepare_visualization_data(self, eval_data, model_outputs, feat_name):
        """Prepare visualization data.
        
        Args:
            eval_data (dict): Evaluation data
            model_outputs (dict): Model outputs
            feat_name (list): Feature name list
            
        Returns:
            dict: Organized visualization data
        """
        x_inverse = eval_data['x_inverse']
        out_feats = model_outputs['out_feats']
        weights = model_outputs['weights']
        bias = model_outputs['bias']
        
        # 组织不同阶数的数据
        order_data = {}
        feature_names = []
        weights_list = []
        start_idx = 0
        
        for order in self.rop_model._nary_indices.keys():
            order_data[order] = {}
            
            for subnet in range(self.rop_model._num_subnets):
                # 获取当前阶数的输入特征
                nary_indices = self.rop_model._nary_indices[order]
                x_features = x_inverse[:, nary_indices]
                
                # 获取对应的输出特征
                num_features = x_features.shape[1]
                z_features = out_feats[:, start_idx:start_idx + num_features].unsqueeze(-1)
                
                # 获取对应的权重
                feature_weights = weights[start_idx:start_idx + num_features]
                
                # 组合输入和输出特征
                combined_features = torch.cat((x_features, z_features), dim=-1)
                order_data[order][subnet] = combined_features.cpu().numpy()
                
                # 更新索引和权重列表
                start_idx += num_features
                weights_list.extend(feature_weights)
                
                # 构建特征名称
                self._build_feature_names(nary_indices, feat_name, feature_names)
        
        return {
            'order_data': order_data,
            'feature_names': feature_names,
            'weights_list': weights_list,
            'num_features': out_feats.shape[1],
            'bias': bias
        }
    
    def _build_feature_names(self, nary_indices, feat_name, feature_names):
        """Build feature name list.
        
        Args:
            nary_indices: N-ary feature indices
            feat_name (list): Original feature names
            feature_names (list): List to populate with feature names
        """
        if feat_name is None:
            # If no feature names provided, use default names
            for i, indices in enumerate(nary_indices):
                if len(indices) > 1:
                    feature_names.append([f"Feature_{idx}" for idx in indices])
                else:
                    feature_names.append([f"Feature_{indices[0]}"])
        else:
            # Use provided feature names
            for i, indices in enumerate(nary_indices):
                try:
                    if len(indices) > 1:
                        feature_names.append([feat_name[idx] for idx in indices])
                    else:
                        feature_names.append([feat_name[indices[0]]])
                except IndexError:
                    print(f"Warning: feature index {indices} out of feat_name range, using default names")
                    feature_names.append([f"Feature_{idx}" for idx in indices])
    
    def _create_and_save_plots(self, viz_data, save_dir, grid_resolution, num_target_at_end = 1):
        """Create and save visualization plots.
        
        Args:
            viz_data (dict): Visualization data
            save_dir (str): Save directory
            grid_resolution (int): Grid resolution
            num_target_at_end (int): Number of target inputs appended at the end of all inputs
        """
        # Compute subplot layout
        n_features = viz_data['num_features']
        n_cols = 3
        n_rows = math.ceil(n_features / n_cols)
        
        # Set figure size - leave more space for colorbar and labels
        subplot_size = 5
        fig_width = n_cols * subplot_size * 1.4
        fig_height = n_rows * subplot_size * 1.1
        
        # Create figure with better subplot spacing
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Plot charts for different orders
        plot_index = 0
        for order in viz_data['order_data'].keys():
            plot_index = self._plot_order_data(
                viz_data, order, fig, plot_index, grid_resolution, num_target_at_end, n_rows, n_cols
            )
        
        # Adjust layout, increase subplot spacing
        plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)
        
        bias = viz_data['bias'].item()
        
        if 'final_output' in viz_data:
            final_output = viz_data['final_output'].item()
            title = f"Bias: {bias:.4f}, Final Output: {final_output:.4f}"
        else:
            title = f"Bias: {bias:.4f} (Global Explanation)"
            
        plt.title(title)
        # # Add title in the upper right corner of the figure
        # ax = fig.add_subplot(1, 1, 1)
        # ax.text(0.95, 0.95, title, 
        #         ha='right', va='top', 
        #         transform=ax.transAxes, 
        #         fontsize=12, 
        #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='gray')
        # )
        
        # Ensure save directory exists

        # Save figure
        # save_path = os.path.join(save_dir, 'explanation.png')
        save_path = save_dir
        # Ensure parent directory exists (save_path is a file path, not a directory)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Interpretability chart saved to: {save_path}")
    
    def _plot_order_data(self, viz_data, order, fig, plot_index, grid_resolution, num_target_at_end, n_rows, n_cols):
        """Plot data for a specific order.
        
        Args:
            viz_data (dict): Visualization data
            order (str): Feature order
            fig: matplotlib figure object
            plot_index (int): Current plot index
            grid_resolution (int): Grid resolution
            num_target_at_end (int): Number of target inputs appended at the end of all inputs
            n_rows (int): Number of rows
            n_cols (int): Number of columns
            
        Returns:
            int: Updated plot index
        """
        order_data = viz_data['order_data'][order]
        feature_names = viz_data['feature_names']
        weights_list = viz_data['weights_list']
        
        for subnet in order_data.keys():
            subnet_data = order_data[subnet]
            
            if order == '1':
                # Plot 1-D feature line charts
                plot_index = self._plot_1d_features(
                    subnet_data, fig, plot_index, feature_names, weights_list, num_target_at_end, n_rows, n_cols
                )
            elif order == '2':
                # Plot 2-D feature heatmaps
                plot_index = self._plot_2d_features(
                    subnet_data, fig, plot_index, feature_names, 
                    weights_list, grid_resolution, num_target_at_end, n_rows, n_cols
                )
        
        return plot_index
    
    def _plot_1d_features(self, data, fig, plot_index, feature_names, weights_list, num_target_at_end, n_rows, n_cols):
        """Plot 1-D feature charts.
        
        Args:
            data (np.ndarray): Feature data
            fig: matplotlib figure object
            plot_index (int): Current plot index
            feature_names (list): 特征名称列表
            weights_list (list): Weight list
            
        Returns:
            int: Updated plot index
        """
        

        for feat_idx in range(data.shape[1]):
            # Create subplot
            weight = weights_list[plot_index]

            ax = fig.add_subplot(n_rows, n_cols, plot_index + 1)
            
            # Extract X and Y data
            if num_target_at_end > 0:
                X = data[:-num_target_at_end, feat_idx, 0]
                Y = data[:-num_target_at_end, feat_idx, 1]

                target_X = data[-num_target_at_end:, feat_idx, 0]
                target_Y = data[-num_target_at_end:, feat_idx, 1]

                if weight < 0:
                    target_Y = -target_Y
                    weight = -weight
                    Y = -Y
            else:
                X = data[:, feat_idx, 0]
                Y = data[:, feat_idx, 1]

                if weight < 0:
                    weight = -weight
                    Y = -Y


            # Get feature name
            if plot_index < len(feature_names):
                x_name = feature_names[plot_index][0] if feature_names[plot_index] else 'Input'
            else:
                x_name = 'Input'
            
            # Sort by X for smooth curves; bit diameter (BD) is discrete, use scatter
            sort_idx = np.argsort(X)
            X_sorted = X[sort_idx]
            Y_sorted = Y[sort_idx]
            is_bd = x_name in ("BD", "BIT_DIAMETER")
            if is_bd:
                ax.scatter(
                    X_sorted,
                    Y_sorted,
                    c="blue",
                    s=32,
                    alpha=0.65,
                    edgecolors="navy",
                    linewidths=0.35,
                )
            else:
                ax.plot(X_sorted, Y_sorted, c='blue', linewidth=3.5)
            
            # Set suitable subplot aspect ratio (not forced square, keep data readable)
            ax.set_aspect('auto')
            
            if num_target_at_end > 0:
                ax.scatter(target_X, target_Y, c='red', s=50, edgecolors='black', linewidths=2.5, zorder=5)
                # Draw two dashed lines (vertical and horizontal) for clear visual reference
                ax.axvline(x=target_X, color='black', linestyle='--', linewidth=1.0)
                ax.axhline(y=target_Y, color='black', linestyle='--', linewidth=1.0)

                
                # Annotate the specific x and y values at a suitable position
                target_X = target_X.item()
                target_Y = target_Y.item()
                
                # Calculate chart range to determine optimal label position
                x_range = X_sorted.max() - X_sorted.min()
                y_range = Y_sorted.max() - Y_sorted.min()
                
                # Smart label position selection based on point location
                # If point is in left half, place label on right; if in right half, on left
                x_center = (X_sorted.max() + X_sorted.min()) / 2
                y_center = (Y_sorted.max() + Y_sorted.min()) / 2
                
                if target_X < x_center:
                    # Point on left, place label on upper right
                    label_x = target_X + x_range * 0.2
                    ha = 'left'
                else:
                    # Point on right, place label on upper left
                    label_x = target_X - x_range * 0.2
                    ha = 'right'
                
                if target_Y < y_center:
                    # Point on bottom, place label above
                    label_y = target_Y + y_range * 0.2
                    va = 'bottom'
                else:
                    # Point on top, place label below
                    label_y = target_Y - y_range * 0.2
                    va = 'top'
                
                # Add text label with background
                ax.text(
                    label_x, label_y, 
                    f'{x_name}={target_X:.2f}, Contribution={target_Y:.2f}', 
                    fontsize=9,
                    ha=ha, 
                    va=va,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='gray')
                )

            ax.set_xlabel(x_name)
            ax.set_ylabel('Output')
            
            # Set title (including weight info)
            if plot_index < len(weights_list):
                ax.set_title(f'Weight: {weight:.4f}')
            
            ax.grid(True, alpha=0.3)
            plot_index += 1
        
        return plot_index
    
    def _plot_2d_features(self, data, fig, plot_index, feature_names, 
                         weights_list, grid_resolution, num_target_at_end, n_rows, n_cols):
        """Plot 2-D feature heatmaps.
        
        Args:
            data (np.ndarray): Feature data
            fig: matplotlib figure object
            plot_index (int): Current plot index
            feature_names (list): 特征名称列表
            weights_list (list): Weight list
            grid_resolution (int): Grid resolution
            
        Returns:
            int: Updated plot index
        """
        for feat_idx in range(data.shape[1]):
            # Create subplot
            weight = weights_list[plot_index]

            ax = fig.add_subplot(n_rows, n_cols, plot_index + 1)
            
            # Extract X, Y, Z data
            if num_target_at_end > 0:
                X = data[:-num_target_at_end, feat_idx, 0]
                Y = data[:-num_target_at_end, feat_idx, 1]
                Z = data[:-num_target_at_end, feat_idx, 2]

                target_X = data[-num_target_at_end:, feat_idx, 0]
                target_Y = data[-num_target_at_end:, feat_idx, 1]
                target_Z = data[-num_target_at_end:, feat_idx, 2]

                if weight < 0:
                    target_Z = -target_Z
                    weight = -weight
                    Z = -Z


            else:
                X = data[:, feat_idx, 0]
                Y = data[:, feat_idx, 1]
                Z = data[:, feat_idx, 2]

                if weight < 0:
                    weight = -weight
                    Z = -Z
            
            # Get feature name
            if plot_index < len(feature_names) and len(feature_names[plot_index]) >= 2:
                x_name = feature_names[plot_index][0]
                y_name = feature_names[plot_index][1]
            else:
                x_name = 'Feature 1'
                y_name = 'Feature 2'
            
            # Create heatmap
            success = self._create_heatmap(
                ax, X, Y, Z, x_name, y_name, grid_resolution
            )
            
            # Set suitable aspect ratio (not forced equal, avoid distortion)
            ax.set_aspect('auto')
            
            # If target points exist, add red markers
            if num_target_at_end > 0:
                # Add red scatter points on heatmap
                ax.scatter(target_X, target_Y, c='red', s=100, edgecolors='black', linewidths=2, zorder=5)
                
                # Add dashed guidelines
                ax.axvline(x=target_X, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
                ax.axhline(y=target_Y, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
                
                # Get specific values
                target_X_val = target_X.item()
                target_Y_val = target_Y.item()
                target_Z_val = target_Z.item()
                
                # Calculate chart range for optimal label position
                x_range = X.max() - X.min()
                y_range = Y.max() - Y.min()
                
                # Smart label position selection based on point location，增加偏移量避免重叠
                x_center = (X.max() + X.min()) / 2
                y_center = (Y.max() + Y.min()) / 2
                
                if target_X_val < x_center:
                    # Point on left, place label on right
                    label_x = target_X_val + x_range * 0.25
                    ha = 'left'
                else:
                    # Point on right, place label on left
                    label_x = target_X_val - x_range * 0.25
                    ha = 'right'
                
                if target_Y_val < y_center:
                    # Point on bottom, place label above
                    label_y = target_Y_val + y_range * 0.25
                    va = 'bottom'
                else:
                    # Point on top, place label below
                    label_y = target_Y_val - y_range * 0.25
                    va = 'top'
                
                # Add text label with background，包含x,y,z三个值
                ax.text(
                    label_x, label_y,
                    f'{x_name}={target_X_val:.2f}\n{y_name}={target_Y_val:.2f}\nContribution={target_Z_val:.2f}',
                    fontsize=8,
                    ha=ha,
                    va=va,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='gray'),
                    zorder=6
                )
            
            # 设置标题
            if plot_index < len(weights_list):
                ax.set_title(f'Weight: {weight:.4f}')
            
            plot_index += 1
        
        return plot_index
    
    def _create_heatmap(self, ax, X, Y, Z, x_name, y_name, grid_resolution):
        """Create heatmap.
        
        Args:
            ax: matplotlib axes object
            X, Y, Z (np.ndarray): Data points
            grid_resolution (int): Grid resolution
            
        Returns:
            bool: Whether heatmap was created successfully
        """
        _bd = frozenset({"BD", "BIT_DIAMETER"})
        try:
            if _bd.intersection({str(x_name), str(y_name)}):
                scatter = ax.scatter(
                    X, Y, c=Z, cmap="viridis", alpha=0.65, s=28, edgecolors="none"
                )
                cbar = plt.colorbar(scatter, ax=ax, label="Output", shrink=0.8, aspect=20)
                cbar.ax.tick_params(labelsize=8)
                ax.set_xlabel(x_name)
                ax.set_ylabel(y_name)
                return True

            # Create uniform grid
            xi = np.linspace(np.min(X), np.max(X), grid_resolution)
            yi = np.linspace(np.min(Y), np.max(Y), grid_resolution)
            Xi, Yi = np.meshgrid(xi, yi)
            
            # Use nearest-neighbor interpolation for Z values on grid
            Zi = griddata((X, Y), Z, (Xi, Yi), method='nearest')
            
            # Draw heatmap
            im = ax.pcolormesh(Xi, Yi, Zi, cmap='viridis', shading='auto')
            
            # Create colorbar with suitable size and position
            cbar = plt.colorbar(im, ax=ax, label='Output', shrink=0.8, aspect=20)
            cbar.ax.tick_params(labelsize=8)
            
            # Add contour lines
            contour = ax.contour(Xi, Yi, Zi, colors='k', linewidths=1.2, alpha=0.5)
            
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            
            return True
            
        except Exception as e:
            print(f"Heatmap creation failed, using scatter plot instead: {e}")
            # Fallback: use scatter plot
            scatter = ax.scatter(X, Y, c=Z, cmap='viridis', alpha=0.6)
            cbar = plt.colorbar(scatter, ax=ax, label='Output', shrink=0.8, aspect=20)
            cbar.ax.tick_params(labelsize=8)
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            
            return False

