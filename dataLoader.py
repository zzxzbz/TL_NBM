from torch.utils.data import Dataset
import pandas as pd

from sklearn.preprocessing import StandardScaler
import torch 
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class ContinuousLearningROPDataset(Dataset):
    """
    Dataset for continuous learning scenario.
    Loads mud logging data by well, and predicts ROP for each row using that row's features.
    """
    def __init__(self, df, scaler = None):

        # df is already a DataFrame, use reference directly to avoid unnecessary copy
        self.all_data = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)

        # Model input features
        self.input_col = ['DMEA','WOBA', 'RPMA', 'MFIA', 'MDIA','SPPA', 'BIT_DIAMETER', 'ANGLE','TQA_ON_BIT']
        self.input_feat = self.all_data[self.input_col]

        # Target variable
        self.target_col = ['ROPA']#,'ROPACAL','ROPA'

        # Data standardization: fit in-place if no scaler provided
        self.scaler = scaler
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.input_feat)

        # One-shot transform, convert to float32 numpy, avoid repeated conversions
        self.scalered_input_feat = self.scaler.transform(self.input_feat).astype(np.float32, copy=False)

        self.input_df_normalized_min = self.scalered_input_feat.min(axis=0)
        self.input_df_normalized_max = self.scalered_input_feat.max(axis=0)

        self.output_feat = self.all_data[self.target_col]

        # Pre-convert entire input/target to torch.Tensor; __getitem__ only does index return
        self._x = torch.from_numpy(self.scalered_input_feat)
        self._y = torch.from_numpy(self.output_feat.values.astype(np.float32, copy=False))

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        return {'input': self._x[idx], 'target': self._y[idx]}
   
class AllForOneROPDataset(Dataset):
    """
    Dataset for all-for-one (cross-well) scenario.
    Loads mud logging data from multiple wells and predicts ROP for each row.
    """
    def __init__(self, file_path_list, scaler = None):
        self.df = [pd.read_csv(file_path, index_col=0) for file_path in file_path_list]

        self.all_data = pd.concat(self.df)
        
        # Model input features
        self.input_col = ['DMEA','WOBA', 'RPMA', 'MFIA', 'MDIA','SPPA', 'BIT_DIAMETER', 'ANGLE','TQA_ON_BIT']
        self.input_feat = self.all_data[self.input_col]
        
        self.target_col = ['ROPA']#,'ROPACAL','ROPA'
        
        self.scaler = scaler
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.input_feat)
        
        self.scalered_input_feat = self.scaler.transform(self.input_feat)

        self.input_df_normalized_min = self.scalered_input_feat.min(axis=0)
        self.input_df_normalized_max = self.scalered_input_feat.max(axis=0)
        
        self.output_feat = self.all_data[self.target_col]

    def __len__(self):
        return self.all_data.shape[0]
    
    def __getitem__(self, idx):
        
        # set torch.float32
        return {'input': torch.tensor(self.scalered_input_feat[idx],dtype=torch.float32),
                'target': torch.tensor(self.output_feat.iloc[idx].values, dtype=torch.float32)}

class AllForOneROPDataset_AE(Dataset):
    """
    Dataset for all-for-one (cross-well) scenario.
    """
    def __init__(self, file_path_list, scaler = None):
        self.df = [pd.read_csv(file_path, index_col=0) for file_path in file_path_list]

        self.all_data = pd.concat(self.df)
        
        # Model input features
        self.input_col = ['DMEA','WOBA', 'RPMA', 'MFIA', 'MDIA','SPPA', 'BIT_DIAMETER', 'ANGLE','TQA_ON_BIT']
        self.input_feat = self.all_data[self.input_col]
        
        self.target_col = ['ROPA']#,'ROPACAL','ROPA'
        
        self.scaler = scaler
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.input_feat)
        
        self.scalered_input_feat = self.scaler.transform(self.input_feat)

        self.input_df_normalized_min = self.scalered_input_feat.min(axis=0)
        self.input_df_normalized_max = self.scalered_input_feat.max(axis=0)
        
        self.output_feat = self.all_data[self.target_col]

    def __len__(self):
        return self.all_data.shape[0]
    
    def __getitem__(self, idx):
        
        # set torch.float32
        return {'input': torch.tensor(self.scalered_input_feat[idx],dtype=torch.float32),
                'target': torch.tensor(self.output_feat.iloc[idx].values, dtype=torch.float32)}
    
class ContinuousLearningROPDataset_BE(Dataset):
    """
    Dataset for continuous learning scenario, predicting BE instead of ROPA.
    """
    def __init__(self, df, scaler = None):

        self.all_data = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)

        # Model input features
        self.input_col = ['DMEA','WOBA', 'RPMA', 'MFIA', 'MDIA','SPPA', 'BIT_DIAMETER', 'ANGLE','TQA_ON_BIT']
        self.input_feat = self.all_data[self.input_col]

        # Target variable
        self.target_col = ['BE']

        self.scaler = scaler
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.input_feat)

        self.scalered_input_feat = self.scaler.transform(self.input_feat).astype(np.float32, copy=False)

        self.input_df_normalized_min = self.scalered_input_feat.min(axis=0)
        self.input_df_normalized_max = self.scalered_input_feat.max(axis=0)

        self.output_feat = self.all_data[self.target_col]

        self._x = torch.from_numpy(self.scalered_input_feat)
        self._y = torch.from_numpy(self.output_feat.values.astype(np.float32, copy=False))

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        return {'input': self._x[idx], 'target': self._y[idx]}

class ContinuousLearningROPDataset_NE(Dataset):
    """
    Dataset for continuous learning scenario, predicting NE instead of ROPA.
    """
    def __init__(self, df, scaler = None):

        self.all_data = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)

        # Model input features
        self.input_col = ['DMEA','WOBA', 'RPMA', 'MFIA', 'MDIA','SPPA', 'BIT_DIAMETER', 'ANGLE','TQA_ON_BIT']
        self.input_feat = self.all_data[self.input_col]

        # Target variable
        self.target_col = ['NE']

        self.scaler = scaler
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.input_feat)

        self.scalered_input_feat = self.scaler.transform(self.input_feat).astype(np.float32, copy=False)

        self.input_df_normalized_min = self.scalered_input_feat.min(axis=0)
        self.input_df_normalized_max = self.scalered_input_feat.max(axis=0)

        self.output_feat = self.all_data[self.target_col]

        self._x = torch.from_numpy(self.scalered_input_feat)
        self._y = torch.from_numpy(self.output_feat.values.astype(np.float32, copy=False))

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        return {'input': self._x[idx], 'target': self._y[idx]}
