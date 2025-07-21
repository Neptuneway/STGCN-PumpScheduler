"""
Data loader module for STGCN head loss prediction
Handles data loading, preprocessing, and transformation
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import csc_matrix
from sklearn import preprocessing
from config import Config


def load_adjacency_matrix(dataset_name):
    """
    Load adjacency matrix for the specified dataset
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        tuple: (adjacency_matrix, number_of_vertices)
    """
    dataset_path = os.path.join(Config.DATA_PATH, dataset_name)
    adj_file_path = os.path.join(dataset_path, Config.ADJACENCY_MATRIX_FILE)
    
    # Load adjacency matrix
    adj = pd.read_csv(adj_file_path)
    adj = adj.iloc[:, 1:]  # Remove index column
    adj = adj.values
    adj = csc_matrix(adj)
    
    # Get number of vertices based on dataset
    if dataset_name == 'metr-la':
        n_vertex = 207
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'pemsd7-m':
        n_vertex = 228
    elif dataset_name == 'tanggu':
        n_vertex = Config.NUM_MONITORING_NODES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return adj, n_vertex


def load_dataset(dataset_name, len_train, len_val, file_name):
    """
    Load and split dataset into train, validation, and test sets
    
    Args:
        dataset_name (str): Name of the dataset
        len_train (int): Length of training set
        len_val (int): Length of validation set
        file_name (str): Name of the data file
        
    Returns:
        tuple: (train_data, validation_data, test_data)
    """
    dataset_path = os.path.join(Config.DATA_PATH, dataset_name)
    file_path = os.path.join(dataset_path, file_name)
    
    # Load data with specified encoding
    data = pd.read_csv(file_path, encoding=Config.DATA_ENCODING)
    
    # Split data
    train = data[:len_train]
    val = data[len_train:len_train + len_val]
    test = data[len_train + len_val:]
    
    return train, val, test


def transform_data(data, n_his, n_pred, device, batch_size):
    """
    Transform raw data into training format for STGCN
    
    Args:
        data (np.ndarray): Raw data array
        n_his (int): Number of historical time steps
        n_pred (int): Number of prediction time steps
        device (torch.device): Device to place tensors on
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (input_tensor, target_tensor)
    """
    # Get dimensions
    n_vertex = data.shape[1]
    x_num = Config.NUM_MONITORING_NODES  # Number of monitoring nodes (input features)
    y_num = Config.NUM_HEAD_LOSS_NODES   # Number of head loss nodes (target features)
    
    # Calculate number of samples
    len_record = len(data)
    num_samples = len_record - n_his - n_pred
    
    # Initialize arrays
    x = np.zeros([num_samples, 1, n_his, x_num])
    y = np.zeros([num_samples, y_num])
    
    # Split input and target data
    x_data = data[:, :x_num]  # Monitoring data (input)
    y_data = data[:, x_num:]  # Head loss data (target)
    
    # Create sliding window samples
    for i in range(num_samples):
        head = i
        tail = i + n_his
        
        # Input: historical monitoring data
        x[i, :, :, :] = x_data[head:tail].reshape(1, n_his, x_num)
        
        # Target: future head loss at prediction time, selecting only the first 'y_num' columns
        y[i] = y_data[tail + n_pred - 1, :y_num]
    
    print(f'Input shape: {x.shape}')
    print(f'Target shape: {y.shape}')
    
    # Convert to PyTorch tensors
    x_tensor = torch.Tensor(x).to(device)
    y_tensor = torch.Tensor(y).to(device)
    
    return x_tensor, y_tensor


def load_adj(dataset_name):
    """
    Legacy function for backward compatibility
    """
    return load_adjacency_matrix(dataset_name)


def load_data(dataset_name, len_train, len_val, file_name):
    """
    Legacy function for backward compatibility
    """
    return load_dataset(dataset_name, len_train, len_val, file_name)


def data_transform(data, n_his, n_pred, device, batch_size):
    """
    Legacy function for backward compatibility
    """
    return transform_data(data, n_his, n_pred, device, batch_size) 
    

