import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

def load_and_preprocess_data(file_path):
    """Load and preprocess a single Excel file"""
    df = pd.read_excel(file_path)
    
    # Get only the normalized columns (those ending with '_normalized')
    norm_cols = [col for col in df.columns if col.endswith('_normalized')]
    norm_cols.sort()
    
    # Extract the data matrix
    data = df[norm_cols].values.T  # Shape: (7, wavelength_points)
    
    # Check for NaN or inf values
    if np.isnan(data).any() or np.isinf(data).any():
        #print(f"\nFound NaN/inf in file: {os.path.basename(file_path)}")
        
        # Fill NaN values with previous timestep values
        for i in range(1, len(data)):  # Start from second timestep
            mask = np.isnan(data[i])
            if mask.any():  # If there are any NaN values in this timestep
                #print(f"Filling NaN values in timestep {i} with values from timestep {i-1}")
                data[i][mask] = data[i-1][mask]
        
        # Check if first timestep has NaN values
        if np.isnan(data[0]).any():
            print("Filling NaN values in first timestep with zeros")
            data[0][np.isnan(data[0])] = 0.0
        
        # Final check for any remaining NaN values
        if np.isnan(data).any():
            print("Warning: Some NaN values could not be filled with previous timesteps")
            # Fill remaining NaN values with 0
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    
    return data

def extract_label(filename):
    """Extract label from filename"""
    # Filename format: label_number_well_*.xlsx
    label = int(filename.split('_')[0])
    return label

class SpectrometerDataset(Dataset):
    def __init__(self, data_dir='./separated_wells_norm', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.file_indices = {}
        
        # Load all files and their labels
        for filename in os.listdir(data_dir):
            if filename.endswith('.xlsx'):
                file_path = os.path.join(data_dir, filename)
                try:
                    # Load and preprocess the data
                    data = load_and_preprocess_data(file_path)
                    
                    # Store the index of this file
                    current_idx = len(self.samples)
                    self.file_indices[current_idx] = filename
                    
                    # Extract label from filename
                    label = extract_label(filename)
                    
                    # Verify data shape
                    if data.shape[0] != 7:
                        print(f"Warning: Unexpected number of timesteps in {filename}: {data.shape[0]}")
                        continue
                    
                    # Store the data and label
                    self.samples.append((data, label))
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        print(f"\nTotal samples loaded: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data, label = self.samples[idx]
        
        # Final safety check
        if np.isnan(data).any():
            print(f"Warning: NaN values still present in {self.file_indices[idx]}")
            # Fill any remaining NaN values with previous timestep values
            for i in range(1, len(data)):
                mask = np.isnan(data[i])
                if mask.any():
                    data[i][mask] = data[i-1][mask]
            # Fill any remaining NaN values with 0
            data = np.nan_to_num(data, nan=0.0)
        
        # Convert to torch tensors
        data = torch.FloatTensor(data)
        label = torch.FloatTensor([label]).squeeze()
        
        if self.transform:
            data = self.transform(data)
            
        return data, label

def create_data_loaders(data_dir='./separated_wells_norm', batch_size=32, train_split=0.8, val_split=0.1):
    """Create train, validation, and test data loaders"""
    
    # Create full dataset
    dataset = SpectrometerDataset(data_dir)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Create indices for the splits
    indices = list(range(total_size))
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Print files in each split that contain NaN values
    print("\nChecking for NaN values in splits:")
    for split_name, split_indices in [("Train", train_indices), 
                                    ("Validation", val_indices), 
                                    ("Test", test_indices)]:
        print(f"\n{split_name} split:")
        for idx in split_indices:
            data, _ = dataset[idx]
            if torch.isnan(data).any():
                print(f"NaN found in {dataset.file_indices[idx]}")
    
    # Create data loaders               what is subsetrandomsampler?
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                            sampler=torch.utils.data.SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=batch_size,
                          sampler=torch.utils.data.SubsetRandomSampler(val_indices))
    test_loader = DataLoader(dataset, batch_size=batch_size,
                           sampler=torch.utils.data.SubsetRandomSampler(test_indices))
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the data loader
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Print some statistics
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Check the shape of one batch
    for batch_data, batch_labels in train_loader:
        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        print(f"Sample labels: {batch_labels[:5]}")
        break
