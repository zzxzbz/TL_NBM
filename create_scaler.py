#!/usr/bin/env python3
"""
Create and save the unified feature scaler.
"""

import pickle
import os
from dataset.dataLoader import AllForOneROPDataset
# python create_scaler.py
def create_and_save_scaler():
    """Create the unified scaler and save it."""
    print("Creating unified scaler...")
    '''
    # All training well IDs
    well_ids = ['B2','B3','B4','B6','B7','B8','B9','B10',' B11','B13','B14',
                'B15','B16H','B17H','B18','B19','B20H','B22H','B24H'] 
    val_well_ids = ['B4','B15', 'B22H','B18H']
    test_well_ids = ['B2','B3']
    '''
    well_ids = ['V0', 'V1', 'V2', 'V3', 'V4',
                'V5', 'V6']
    val_well_ids = ['V3']
    test_well_ids = ['V2']
    
    # Training set well IDs
    train_well_ids = [well_id for well_id in well_ids if well_id not in val_well_ids and well_id not in test_well_ids]
    
    # Training set file paths
    train_well_file_path_list = [f'./data/ProcessedData-2/{well_id}.csv' for well_id in train_well_ids]
    
    print(f"   Number of training wells: {len(train_well_ids)}")
    print(f"   Training well IDs: {train_well_ids}")
    
    # Create dataset (this automatically creates the scaler)
    train_dataset = AllForOneROPDataset(train_well_file_path_list)
    scaler = train_dataset.scaler
    
    print(f"   Input features: {train_dataset.input_col}")
    print(f"   Number of features: {len(train_dataset.input_col)}")
    
    # Create save directory
    scaler_dir = './scalers'
    os.makedirs(scaler_dir, exist_ok=True)
    
    # Save scaler
    scaler_path = f'{scaler_dir}/unified_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"   [OK] Scaler saved: {scaler_path}")
    
    # Save feature column name info
    feature_info = {
        'input_columns': train_dataset.input_col,
        'target_columns': train_dataset.target_col,
        'input_dim': len(train_dataset.input_col),
        'train_wells': train_well_ids,
        'val_wells': val_well_ids,
        'test_wells': test_well_ids
    }
    
    feature_info_path = f'{scaler_dir}/feature_info.pkl'
    with open(feature_info_path, 'wb') as f:
        pickle.dump(feature_info, f)
    
    print(f"   [OK] Feature info saved: {feature_info_path}")
    
    return scaler_path, feature_info_path

def load_scaler(verbose: bool = True):
    """Load the unified scaler."""
    scaler_path = './scalers/unified_scaler.pkl'
    feature_info_path = './scalers/feature_info.pkl'
    
    if not os.path.exists(scaler_path):
        if verbose:
            print("[ERROR] Scaler file not found, please run create_and_save_scaler() first")
        return None, None
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load feature info
    with open(feature_info_path, 'rb') as f:
        feature_info = pickle.load(f)
    
    if verbose:
        print(f"[OK] Scaler loaded: {scaler_path}")
        print(f"   Input features: {feature_info['input_columns']}")
        print(f"   Feature dimension: {feature_info['input_dim']}")
    
    return scaler, feature_info

def main():
    """Main function"""
    print("Creating unified scaler")
    print("=" * 30)
    
    try:
        # Create and save scaler
        scaler_path, feature_info_path = create_and_save_scaler()
        
        print(f"\nScaler creation complete!")
        print(f"File save locations:")
        print(f"   - {scaler_path}")
        print(f"   - {feature_info_path}")
        
        # Test loading
        print(f"\nTesting load...")
        scaler, feature_info = load_scaler()
        
        if scaler is not None:
            print("[OK] Scaler load test passed!")
        else:
            print("[ERROR] Scaler load test failed!")
            
    except Exception as e:
        print(f"[ERROR] Creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    # python create_scaler.py
