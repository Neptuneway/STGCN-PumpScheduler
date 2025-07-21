"""
STGCN Head Loss Prediction Main Script
Spatial-Temporal Graph Convolutional Network for Water Distribution Network Head Loss Prediction

This script implements STGCN to predict head loss at 8 monitoring points
using data from 30 monitoring points in a water distribution network.
"""

import logging
import os
import argparse
import math
import random
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import matplotlib.pyplot as plt
import tqdm
from sklearn import preprocessing

from script import dataloader, utility, earlystopping
from model import models_eval
from config import Config

# Clear GPU cache
torch.cuda.empty_cache()


def set_environment(seed):
    """
    Set up environment for reproducible results
    
    Args:
        seed (int): Random seed for reproducibility
    """
    # Set available CUDA devices
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_parameters():
    """
    Parse command line arguments and return configuration
    
    Returns:
        tuple: (args, device, blocks)
    """
    parser = argparse.ArgumentParser(description='STGCN Head Loss Prediction')
    
    # System parameters
    parser.add_argument('--enable_cuda', type=bool, default=Config.ENABLE_CUDA, 
                       help='Enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=Config.SEED, 
                       help='Random seed for reproducible results')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default=Config.DATASET_NAME, 
                       choices=['metr-la', 'pems-bay', 'pemsd7-m', 'tanggu'])
    parser.add_argument('--n_his', type=int, default=Config.HISTORICAL_STEPS)
    parser.add_argument('--n_pred', type=int, default=Config.PREDICTION_STEPS, 
                       help='Number of time intervals for prediction')
    parser.add_argument('--time_intvl', type=int, default=Config.TIME_INTERVAL)
    
    # Model architecture parameters
    parser.add_argument('--Kt', type=int, default=Config.KERNEL_SIZE, 
                       help='Temporal kernel size')
    parser.add_argument('--stblock_num', type=int, default=Config.ST_BLOCK_NUM)
    parser.add_argument('--act_func', type=str, default=Config.ACTIVATION_FUNC, 
                       choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=Config.CHEBYSHEV_ORDER, 
                       choices=[3, 2], help='Chebyshev polynomial order')
    parser.add_argument('--graph_conv_type', type=str, default=Config.GRAPH_CONV_TYPE, 
                       choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default=Config.GSO_TYPE, 
                       choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, 
                       help='Enable bias in layers')
    parser.add_argument('--droprate', type=float, default=Config.DROPOUT_RATE)
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, 
                       help='Learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=Config.WEIGHT_DECAY, 
                       help='Weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, 
                       help='Number of training epochs')
    parser.add_argument('--opt', type=str, default=Config.OPTIMIZER, 
                       help='Optimizer choice', choices=['adam', 'rmsprop', 'adamw'])
    parser.add_argument('--step_size', type=int, default=Config.STEP_SIZE)
    parser.add_argument('--gamma', type=float, default=Config.GAMMA)
    parser.add_argument('--patience', type=int, default=Config.PATIENCE, 
                       help='Early stopping patience')
    parser.add_argument('--cleanornot', type=bool, default=Config.CLEAN_OUTPUT, 
                       help='Whether to filter output data')
    
    args = parser.parse_args()
    print(f'Training configurations: {args}')
    
    # Save training configurations
    save_training_configs(args)
    
    # Set environment for reproducible results
    set_environment(args.seed)
    
    # Set device
    device = get_device(args.enable_cuda)
    
    # Calculate output dimension
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
    
    # Get model architecture blocks
    blocks = Config.get_model_blocks()
    
    return args, device, blocks


def save_training_configs(args):
    """Save training configurations to file"""
    config_path = os.path.join(Config.RESULT_PATH, 'training_configs.txt')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write('Training configurations:\n')
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')


def get_device(enable_cuda):
    """
    Get the appropriate device (GPU or CPU)
    
    Args:
        enable_cuda (bool): Whether to enable CUDA
        
    Returns:
        torch.device: Device to use for computation
    """
    if enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print('\nGPU is available and will be used')
    else:
        device = torch.device('cpu')
        print('\nGPU is not available, using CPU')
    
    return device


def filter_custom_data(dataset, filter_list):
    """
    Filter dataset based on custom conditions
    
    Args:
        dataset: PyTorch dataset
        filter_list: List of filter conditions
        
    Returns:
        list: Filtered dataset or None if empty
    """
    filter_condition = lambda m: m == 0
    filtered_batch = []
    
    for i, (x, y) in enumerate(dataset):
        if filter_condition(filter_list[i]):
            filtered_batch.append((x, y))
    
    if not filtered_batch:
        return None
    else:
        print(f'\nFiltered batch size after filtering: {len(filtered_batch)}')
        return filtered_batch


def prepare_data(args, device):
    """
    Prepare data for training, validation, and testing
    
    Args:
        args: Command line arguments
        device: Device to place tensors on
        
    Returns:
        tuple: (n_vertex, scaler, train_iter, val_iter, test_iter, train_original)
    """
    # Load adjacency matrix
    adj, n_vertex = dataloader.load_adj(args.dataset)
    print(f'Adjacency matrix shape: {adj.shape}, Number of vertices: {n_vertex}')
    
    # Calculate graph shift operator
    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray().astype(np.float32)
    args.gso = torch.from_numpy(gso).to(device)
    print('Adjacency matrix loaded successfully')
    
    # Load and split data
    dataset_path = os.path.join(Config.DATA_PATH, args.dataset)
    data_file = Config.TRAIN_DATA_FILE
    data_length = pd.read_csv(os.path.join(dataset_path, data_file), 
                             encoding=Config.DATA_ENCODING).shape[0]
    
    # Calculate split lengths
    val_test_rate = Config.VAL_TEST_RATE
    len_val = int(math.floor(data_length * val_test_rate))
    len_test = int(math.floor(data_length * val_test_rate))
    len_train = int(data_length - len_val - len_test)
    
    # Load raw data
    train_original, val_original, test_original = dataloader.load_data(
        args.dataset, len_train, len_val, data_file)
    
    # Normalize data
    scaler = preprocessing.StandardScaler()
    train_normalized = scaler.fit_transform(train_original)
    val_normalized = scaler.transform(val_original)
    test_normalized = scaler.transform(test_original)
    
    print(f'\nTrain shape: {train_normalized.shape}')
    print(f'Validation shape: {val_normalized.shape}')
    print(f'Test shape: {test_normalized.shape}')
    print(f'Mean: {scaler.mean_}, Standard deviation: {scaler.scale_}')
    
    # Transform data for model input
    test_original_array = test_original.values
    x_test_original, y_test_original = dataloader.data_transform(
        test_original_array, args.n_his, args.n_pred, device, args.batch_size)
    
    x_train, y_train = dataloader.data_transform(
        train_normalized, args.n_his, args.n_pred, device, args.batch_size)
    x_val, y_val = dataloader.data_transform(
        val_normalized, args.n_his, args.n_pred, device, args.batch_size)
    x_test, y_test = dataloader.data_transform(
        test_normalized, args.n_his, args.n_pred, device, args.batch_size)
    
    # Load filter data if needed
    if args.cleanornot:
        filter_file_path = os.path.join(dataset_path, 'output', Config.OUTPUT_FILTER_FILE)
        filter_data = pd.read_csv(filter_file_path, header=None).values
        
        # Split filter data
        train_filter = filter_data[:len(x_train)]
        val_filter = filter_data[len(train_original):len(train_original) + len(x_val)]
        test_filter = filter_data[len(train_original) + len(val_original):len(train_original) + len(val_original) + len(x_test)]
        
        train_filter = torch.tensor(train_filter).view(-1)
        val_filter = torch.tensor(val_filter).view(-1)
        test_filter = torch.tensor(test_filter).view(-1)
    
    # Create data loaders
    train_data = utils.data.TensorDataset(x_train, y_train)
    if args.cleanornot:
        train_data = filter_custom_data(train_data, train_filter)
    train_iter = utils.data.DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    val_data = utils.data.TensorDataset(x_val, y_val)
    if args.cleanornot:
        val_data = filter_custom_data(val_data, val_filter)
    val_iter = utils.data.DataLoader(
        dataset=val_data, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    test_data = utils.data.TensorDataset(x_test, y_test)
    if args.cleanornot:
        test_data = filter_custom_data(test_data, test_filter)
    test_iter = utils.data.DataLoader(
        dataset=test_data, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    return n_vertex, scaler, train_iter, val_iter, test_iter, train_original


def prepare_model(args, blocks, n_vertex, device):
    """
    Prepare model, loss function, optimizer, and scheduler
    
    Args:
        args: Command line arguments
        blocks: Model architecture blocks
        n_vertex: Number of vertices
        device: Device to place tensors on
        
    Returns:
        tuple: (loss_function, early_stopping, model, optimizer, scheduler)
    """
    loss_function = nn.MSELoss()
    early_stopping = earlystopping.EarlyStopping(
        mode='min', min_delta=0.0, patience=args.patience)
    
    # Create model
    if args.graph_conv_type == 'cheb_graph_conv':
        model = models_eval.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    else:
        model = models_eval.STGCNGraphConv(args, blocks, n_vertex).to(device)
    
    # Create optimizer
    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, 
                                 weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, 
                              weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, 
                               weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'Optimizer {args.opt} is not implemented.')
    
    # Create scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma)
    
    return loss_function, early_stopping, model, optimizer, scheduler


def train_model(loss_function, args, optimizer, scheduler, early_stopping, 
                model, train_iter, val_iter, result_path):
    """
    Train the model
    
    Args:
        loss_function: Loss function
        args: Command line arguments
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        early_stopping: Early stopping handler
        model: Model to train
        train_iter: Training data iterator
        val_iter: Validation data iterator
        result_path: Path to save results
    """
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        num_samples = 0
        
        for x, y in tqdm.tqdm(train_iter, desc=f'Epoch {epoch+1}/{args.epochs}'):
            # Forward pass
            y_pred = model(x).view(len(x), -1)
            
            # Calculate loss
            loss = loss_function(y_pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item() * y.shape[0]
            num_samples += y.shape[0]
        
        # Update learning rate
        scheduler.step()
        
        # Validation phase
        val_loss = validate_model(model, val_iter, loss_function)
        train_loss = total_loss / num_samples
        
        train_losses.append(train_loss)
        val_losses.append(val_loss.item())
        
        # Print progress
        gpu_memory = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print(f'Epoch: {epoch+1:03d} | LR: {optimizer.param_groups[0]["lr"]:.6f} | '
              f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | '
              f'GPU Memory: {gpu_memory:.1f} MiB')
        
        # Early stopping
        if early_stopping.step(val_loss):
            print('Early stopping triggered.')
            break
    
    # Save training results
    save_training_results(train_losses, val_losses, result_path)


def validate_model(model, val_iter, loss_function):
    """
    Validate the model
    
    Args:
        model: Model to validate
        val_iter: Validation data iterator
        loss_function: Loss function
        
    Returns:
        torch.Tensor: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for x, y in val_iter:
            y_pred = model(x).view(len(x), -1)
            loss = loss_function(y_pred, y)
            total_loss += loss.item() * y.shape[0]
            num_samples += y.shape[0]
    
    return torch.tensor(total_loss / num_samples)


def save_training_results(train_losses, val_losses, result_path):
    """
    Save training results and plot loss curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        result_path: Path to save results
    """
    # Save loss data
    loss_data = pd.DataFrame({
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_file = os.path.join(result_path, 'training_loss.csv')
    loss_data.to_csv(loss_file, float_format='%.4f')
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, marker='o', linestyle='-', color='blue', label='Training Loss')
    plt.plot(val_losses, marker='^', linestyle='-', color='red', label='Validation Loss')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Time', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(result_path, 'training_loss.png')
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    plt.close()


def test_model(scaler, loss_function, model, test_iter, train_original, args, result_path):
    """
    Test the trained model
    
    Args:
        scaler: Data scaler
        loss_function: Loss function
        model: Trained model
        test_iter: Test data iterator
        train_original: Original training data
        args: Command line arguments
        result_path: Path to save results
    """
    model.eval()
    
    # Evaluate model
    test_mse = utility.evaluate_model(model, loss_function, test_iter)
    test_mae, test_rmse, test_wmape = utility.evaluate_metric(
        model, test_iter, scaler, train_original, result_path)
    
    print(f'Dataset: {args.dataset} | Test MSE: {test_mse:.6f} | '
          f'MAE: {test_mae:.6f} | RMSE: {test_rmse:.6f} | WMAPE: {test_wmape:.8f}')


def main():
    """Main function"""
    # Create result directory
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    result_path = os.path.join(Config.RESULT_PATH, f'predict_result_{current_time}')
    os.makedirs(result_path, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('stgcn')
    
    # Record start time
    start_time = time.time()
    
    try:
        # Get parameters
        args, device, blocks = get_parameters()
        
        # Prepare data
        n_vertex, scaler, train_iter, val_iter, test_iter, train_original = prepare_data(args, device)
        
        # Prepare model
        loss_function, early_stopping, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex, device)
        
        # Train model
        train_model(loss_function, args, optimizer, scheduler, early_stopping, 
                   model, train_iter, val_iter, result_path)
        
        # Save model
        model_path = os.path.join(result_path, 'model.pth')
        torch.save(model.state_dict(), model_path)
        
        # Test model
        test_model(scaler, loss_function, model, test_iter, train_original, args, result_path)
        
        # Record end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        print(f"Results saved to: {result_path}")
        print(f"Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main() 