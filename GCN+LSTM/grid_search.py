import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings
import time
from datetime import datetime
import sys
import itertools
import json
warnings.filterwarnings('ignore')

from model import GCNLSTMModel, EarlyStopping
from data_loader import load_adjacency_matrix, load_training_data, create_data_loaders

class ResultLogger:
    """Result logger, outputs to terminal and file"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        
    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message)
    
    def flush(self):
        self.terminal.flush()

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return np.inf
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def calculate_wmape(y_true, y_pred):
    """Calculate Weighted Mean Absolute Percentage Error (WMAPE)"""
    # Avoid division by zero
    if np.sum(np.abs(y_true)) == 0:
        return np.inf
    
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    return wmape

def train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                num_epochs=100, patience=15, model_save_path='best_model.pth',
                edge_index=None, edge_weight=None, scheduler=None, logger=None):
    """
    Train the model
    """
    model.to(device)
    early_stopping = EarlyStopping(patience=patience)
    
    train_losses = []
    val_losses = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            batch_x = batch_x.to(device)  # [batch_size, seq_len, num_nodes, input_dim]
            batch_y = batch_y.to(device)  # [batch_size, output_dim]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_x, edge_index, edge_weight)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x, edge_index, edge_weight)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            if logger:
                logger.write(f'Epoch {epoch+1}/{num_epochs}:\n')
                logger.write(f'  Train Loss: {avg_train_loss:.6f}\n')
                logger.write(f'  Val Loss: {avg_val_loss:.6f}\n')
                logger.write(f'  Learning Rate: {current_lr:.6f}\n')
        else:
            if logger:
                logger.write(f'Epoch {epoch+1}/{num_epochs}:\n')
                logger.write(f'  Train Loss: {avg_train_loss:.6f}\n')
                logger.write(f'  Val Loss: {avg_val_loss:.6f}\n')
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            if logger:
                logger.write(f'Early stopping triggered after {epoch+1} epochs\n')
            break
    
    training_time = time.time() - start_time
    
    # Save best model
    torch.save(model.state_dict(), model_save_path)
    if logger:
        logger.write(f'Best model saved to {model_save_path}\n')
    
    return train_losses, val_losses, training_time

def evaluate_model(model, test_loader, criterion, device, scaler_head_loss, edge_index=None, edge_weight=None, logger=None):
    """
    Evaluate the model
    """
    model.eval()
    test_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x, edge_index, edge_weight)
            loss = criterion(outputs, batch_y)
            
            test_loss += loss.item()
            
            # Denormalize predictions
            pred_denorm = scaler_head_loss.inverse_transform(outputs.cpu().numpy())
            target_denorm = scaler_head_loss.inverse_transform(batch_y.cpu().numpy())
            
            predictions.extend(pred_denorm)
            targets.extend(target_denorm)
    
    avg_test_loss = test_loss / len(test_loader)
    
    # Calculate evaluation metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # Calculate R² for each output dimension
    r2_scores = []
    mape_scores = []
    mae_scores = []
    
    for i in range(predictions.shape[1]):
        # R² calculation
        ss_res = np.sum((targets[:, i] - predictions[:, i]) ** 2)
        ss_tot = np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        r2_scores.append(r2)
        
        # MAPE calculation
        mape = calculate_mape(targets[:, i], predictions[:, i])
        mape_scores.append(mape)
        
        # MAE calculation
        mae_i = np.mean(np.abs(targets[:, i] - predictions[:, i]))
        mae_scores.append(mae_i)
    
    avg_r2 = np.mean(r2_scores)
    avg_mape = np.mean(mape_scores)
    avg_mae = np.mean(mae_scores)
    
    # Calculate overall WMAPE
    wmape = calculate_wmape(targets, predictions)
    
    if logger:
        logger.write(f'\nTest Results:\n')
        logger.write(f'  Test Loss: {avg_test_loss:.6f}\n')
        logger.write(f'  MSE: {mse:.6f}\n')
        logger.write(f'  RMSE: {rmse:.6f}\n')
        logger.write(f'  MAE: {mae:.6f}\n')
        logger.write(f'  Average R²: {avg_r2:.6f}\n')
        logger.write(f'  Average MAPE: {avg_mape:.6f}%\n')
        logger.write(f'  WMAPE: {wmape:.6f}%\n')
        logger.write(f'  Individual MAPE: {mape_scores}\n')
        logger.write(f'  Individual MAE: {mae_scores}\n')
    
    return {
        'test_loss': avg_test_loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'avg_r2': avg_r2,
        'avg_mape': avg_mape,
        'wmape': wmape,
        'individual_mape': mape_scores,
        'individual_mae': mae_scores,
        'predictions': predictions,
        'targets': targets
    }

def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """Plot training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(results, save_path='predictions.png'):
    """Plot prediction results"""
    predictions = results['predictions']
    targets = results['targets']
    
    # Select first 100 samples for visualization
    n_samples = min(100, len(predictions))
    
    plt.figure(figsize=(15, 10))
    for i in range(9):  # 9 output targets
        plt.subplot(3, 3, i+1)
        plt.scatter(targets[:n_samples, i], predictions[:n_samples, i], alpha=0.6)
        plt.plot([targets[:n_samples, i].min(), targets[:n_samples, i].max()], 
                [targets[:n_samples, i].min(), targets[:n_samples, i].max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'Target {i+1}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_experiment_folder(lstm_layers, lstm_hidden_dim, learning_rate):
    """Create experiment folder"""
    timestamp = datetime.now().strftime("%m%d-%H%M")
    folder_name = f"{timestamp}-{lstm_layers}+{lstm_hidden_dim}+{learning_rate}"
    folder_path = os.path.join(os.getcwd(), folder_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    return folder_path

def single_training_run(config, device, edge_index_device, edge_weight_device, 
                       train_loader, val_loader, test_loader, scaler_head_loss):
    """
    Single training run
    """
    # Create experiment folder
    experiment_folder = create_experiment_folder(
        config['lstm_layers'], 
        config['lstm_hidden_dim'], 
        config['learning_rate']
    )
    
    # Set log file
    log_file = os.path.join(experiment_folder, 'result.txt')
    logger = ResultLogger(log_file)
    
    logger.write(f"=== Grid Search Experiment ===\n")
    logger.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    logger.write(f"Configuration parameters:\n")
    for key, value in config.items():
        logger.write(f"  {key}: {value}\n")
    logger.write(f"Experiment folder: {experiment_folder}\n\n")
    
    # Create model
    model = GCNLSTMModel(
        num_nodes=33,
        input_dim=1,
        hidden_dim=config['hidden_dim'],
        lstm_hidden_dim=config['lstm_hidden_dim'],
        lstm_layers=config['lstm_layers'],
        output_dim=9,
        dropout=config['dropout'],
        enable_bias=True
    )
    
    # Set optimizer and loss function
    optimizer = torch.optim.Adam(  # type: ignore
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=50, 
        gamma=0.5
    )
    
    # Train model
    logger.write("Starting training...\n")
    train_losses, val_losses, training_time = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=config['num_epochs'],
        patience=config['patience'],
        model_save_path=os.path.join(experiment_folder, 'best_model.pth'),
        edge_index=edge_index_device,
        edge_weight=edge_weight_device,
        scheduler=scheduler,
        logger=logger
    )
    
    # Evaluate model
    logger.write("\nStarting evaluation...\n")
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        scaler_head_loss=scaler_head_loss,
        edge_index=edge_index_device,
        edge_weight=edge_weight_device,
        logger=logger
    )
    
    # Add training time to results
    results['training_time'] = training_time
    
    # Plot training curves
    plot_training_curves(
        train_losses, 
        val_losses, 
        save_path=os.path.join(experiment_folder, 'training_curves.png')
    )
    
    # Plot prediction results
    plot_predictions(
        results, 
        save_path=os.path.join(experiment_folder, 'predictions.png')
    )
    
    # Save configuration and results
    config_file = os.path.join(experiment_folder, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    results_file = os.path.join(experiment_folder, 'results.json')
    # Convert numpy arrays to lists for JSON serialization
    results_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()}
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    logger.write(f"\nExperiment completed! Results saved to: {experiment_folder}\n")
    logger.write(f"Training time: {training_time:.2f} seconds\n")
    
    return results, experiment_folder

def main():
    """Main function - Grid Search"""
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Grid search parameters
    learning_rates = [0.01, 0.001, 0.0001, 0.00001]
    lstm_layers_list = [2, 3, 4]
    
    # Fixed parameters
    base_config = {
        'hidden_dim': 512,
        'lstm_hidden_dim': 512,
        'output_dim': 9,
        'dropout': 0.5,
        'batch_size': 288,
        'num_epochs': 300,
        'patience': 20,
        'weight_decay': 0.0005
    }
    
    # Load data
    print("Loading data...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    edge_index, edge_weight = load_adjacency_matrix(os.path.join(base_dir, '取平均结果-exp-108-无向对称.csv'))
    pressure_data, head_loss_data = load_training_data(os.path.join(base_dir, 'train-22and2023-5min.csv'))
    
    # Create data loaders
    train_loader, val_loader, test_loader, scaler_head_loss = create_data_loaders(
        pressure_data, head_loss_data, edge_index, edge_weight,
        seq_length=12,
        target_step=12,
        batch_size=base_config['batch_size']
    )
    
    # Move adjacency matrix to device
    edge_index_device = edge_index.to(device)
    edge_weight_device = edge_weight.to(device) if edge_weight is not None else None
    
    # Store all experiment results
    all_results = []
    
    # Grid search
    total_experiments = len(learning_rates) * len(lstm_layers_list)
    current_experiment = 0
    
    print(f"Starting grid search, total {total_experiments} experiments...")
    
    for lr in learning_rates:
        for lstm_layers in lstm_layers_list:
            current_experiment += 1
            print(f"\n=== Experiment {current_experiment}/{total_experiments} ===")
            print(f"Learning rate: {lr}, LSTM layers: {lstm_layers}")
            
            # Create configuration
            config = base_config.copy()
            config['learning_rate'] = lr
            config['lstm_layers'] = lstm_layers
            
            try:
                # Run single experiment
                results, experiment_folder = single_training_run(
                    config, device, edge_index_device, edge_weight_device,
                    train_loader, val_loader, test_loader, scaler_head_loss
                )
                
                # Save results
                all_results.append({
                    'config': config,
                    'results': results,
                    'experiment_folder': experiment_folder
                })
                
                print(f"Experiment completed: {experiment_folder}")
                
            except Exception as e:
                print(f"Experiment failed: {e}")
                continue
    
    # Generate grid search results summary
    print("\n=== Grid Search Results Summary ===")
    
    # Create summary file
    summary_file = f"grid_search_summary_{datetime.now().strftime('%m%d-%H%M')}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== Grid Search Results Summary ===\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Sort by performance
        sorted_results = sorted(all_results, key=lambda x: x['results']['rmse'])
        
        f.write("Results sorted by RMSE:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<4} {'Learning Rate':<10} {'LSTM Layers':<8} {'RMSE':<10} {'MAE':<10} {'R²':<8} {'Training Time (s)':<12} {'Folder'}\n")
        f.write("-" * 80 + "\n")
        
        for i, result in enumerate(sorted_results):
            config = result['config']
            results_data = result['results']
            folder = result['experiment_folder']
            
            f.write(f"{i+1:<4} {config['learning_rate']:<10} {config['lstm_layers']:<8} "
                   f"{results_data['rmse']:<10.6f} {results_data['mae']:<10.6f} "
                   f"{results_data['avg_r2']:<8.4f} {results_data['training_time']:<12.2f} "
                   f"{os.path.basename(folder)}\n")
        
        f.write("\nBest configuration:\n")
        best_result = sorted_results[0]
        f.write(f"Learning rate: {best_result['config']['learning_rate']}\n")
        f.write(f"LSTM layers: {best_result['config']['lstm_layers']}\n")
        f.write(f"RMSE: {best_result['results']['rmse']:.6f}\n")
        f.write(f"MAE: {best_result['results']['mae']:.6f}\n")
        f.write(f"R²: {best_result['results']['avg_r2']:.4f}\n")
        f.write(f"Training time: {best_result['results']['training_time']:.2f} seconds\n")
        f.write(f"Experiment folder: {best_result['experiment_folder']}\n")
    
    # Print summary to terminal
    print(f"\nGrid search completed! Summary saved to: {summary_file}")
    print(f"Best configuration:")
    best_result = sorted_results[0]
    print(f"  Learning rate: {best_result['config']['learning_rate']}")
    print(f"  LSTM layers: {best_result['config']['lstm_layers']}")
    print(f"  RMSE: {best_result['results']['rmse']:.6f}")
    print(f"  MAE: {best_result['results']['mae']:.6f}")
    print(f"  R²: {best_result['results']['avg_r2']:.4f}")

if __name__ == "__main__":
    main() 