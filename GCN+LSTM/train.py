import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings
import time
from datetime import datetime
import sys
warnings.filterwarnings('ignore')

from model import GCNLSTMModel, EarlyStopping
from data_loader import load_adjacency_matrix, load_training_data, create_data_loaders
from config import CONFIG

class ResultLogger:
    """Result logger, outputs to both terminal and file"""
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
                edge_index=None, edge_weight=None, scheduler=None):
    """
    Train the model
    """
    model.to(device)
    early_stopping = EarlyStopping(patience=patience)
    
    train_losses = []
    val_losses = []
    
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
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.6f}')
            print(f'  Val Loss: {avg_val_loss:.6f}')
            print(f'  Learning Rate: {current_lr:.6f}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.6f}')
            print(f'  Val Loss: {avg_val_loss:.6f}')
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Save best model
    torch.save(model.state_dict(), model_save_path)
    print(f'Best model saved to {model_save_path}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, criterion, device, scaler_head_loss, edge_index=None, edge_weight=None):
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
    
    print(f'\nTest Results:')
    print(f'  Test Loss: {avg_test_loss:.6f}')
    print(f'  MSE: {mse:.6f}')
    print(f'  RMSE: {rmse:.6f}')
    print(f'  MAE: {mae:.6f}')
    print(f'  Average R²: {avg_r2:.6f}')
    print(f'  Average MAPE: {avg_mape:.6f}%')
    print(f'  WMAPE: {wmape:.6f}%')
    
    for i in range(predictions.shape[1]):
        print(f'  Output {i+1}: R²={r2_scores[i]:.6f}, MAPE={mape_scores[i]:.6f}%, MAE={mae_scores[i]:.6f}')
    
    return {
        'test_loss': avg_test_loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_scores': r2_scores,
        'avg_r2': avg_r2,
        'mape_scores': mape_scores,
        'avg_mape': avg_mape,
        'mae_scores': mae_scores,
        'avg_mae': avg_mae,
        'wmape': wmape,
        'predictions': predictions,
        'targets': targets
    }

def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """
    Plot training curves (only save, no display)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close plot, no display

def plot_predictions(results, save_path='predictions.png'):
    """
    Plot prediction results (only save, no display)
    """
    predictions = results['predictions']
    targets = results['targets']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    for i in range(9):
        ax = axes[i]
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.5)
        ax.plot([targets[:, i].min(), targets[:, i].max()], 
                [targets[:, i].min(), targets[:, i].max()], 'r--', lw=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title(f'Output {i+1} (R² = {results["r2_scores"][i]:.3f})')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close plot, no display

def create_experiment_folder(lstm_layers, lstm_hidden_dim, learning_rate, run_id=None):
    """Create experiment folder"""
    now = datetime.now()
    if run_id is not None:
        folder_name = f"{now.strftime('%m%d')}-{now.strftime('%H%M')}-{lstm_layers}+{lstm_hidden_dim}+{learning_rate}-run{run_id}"
    else:
        folder_name = f"{now.strftime('%m%d')}-{now.strftime('%H%M')}-{lstm_layers}+{lstm_hidden_dim}+{learning_rate}"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    return folder_name

def single_training_run(config, device, edge_index_device, edge_weight_device, 
                       train_loader, val_loader, test_loader, scaler_head_loss, run_id):
    """Single training run"""
    print(f"\n{'='*60}")
    print(f"Starting training run {run_id}")
    print(f"{'='*60}")
    
    # Create experiment folder
    experiment_folder = create_experiment_folder(
        config['lstm_layers'], 
        config['lstm_hidden_dim'], 
        config['learning_rate'], 
        run_id
    )
    
    # Set result log file
    log_file = os.path.join(experiment_folder, 'result.txt')
    original_stdout = sys.stdout
    sys.stdout = ResultLogger(log_file)
    
    print(f"Experiment folder: {experiment_folder}")
    print("Configuration parameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Record training start time
    start_time = time.time()
    
    # Create model
    model = GCNLSTMModel(
        num_nodes=config['num_nodes'],
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        lstm_hidden_dim=config['lstm_hidden_dim'],
        lstm_layers=config['lstm_layers'],
        output_dim=config['output_dim'],
        dropout=config['dropout'],
        enable_bias=config['enable_bias']
    )
    
    print(f"\nNumber of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    
    # Optimizer configuration
    optimizer = torch.optim.Adam(model.parameters(),  # type: ignore
                          lr=config['learning_rate'], 
                          weight_decay=config['weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                         step_size=config['step_size'], 
                                         gamma=config['gamma'])

    # Train model
    print("\nStarting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, criterion, device,
        num_epochs=config['num_epochs'],
        patience=config['patience'],
        model_save_path=os.path.join(experiment_folder, 'best_model.pth'),
        edge_index=edge_index_device,
        edge_weight=edge_weight_device,
        scheduler=scheduler
    )
    
    # Calculate training duration
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Plot training curves (only save, no display)
    plot_training_curves(train_losses, val_losses, 
                        save_path=os.path.join(experiment_folder, 'training_curves.png'))
    
    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(model, test_loader, criterion, device, scaler_head_loss,
                            edge_index=edge_index_device, edge_weight=edge_weight_device)
    
    # Plot prediction results (only save, no display)
    plot_predictions(results, save_path=os.path.join(experiment_folder, 'predictions.png'))
    
    # Save detailed results to file
    with open(os.path.join(experiment_folder, 'detailed_results.txt'), 'w', encoding='utf-8') as f:
        f.write("Detailed evaluation results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")
        f.write(f"WMAPE: {results['wmape']:.6f}%\n")
        f.write(f"Average MAPE: {results['avg_mape']:.6f}%\n")
        f.write(f"Average MAE: {results['avg_mae']:.6f}\n")
        f.write(f"Average R²: {results['avg_r2']:.6f}\n")
        f.write(f"MSE: {results['mse']:.6f}\n")
        f.write(f"RMSE: {results['rmse']:.6f}\n")
        f.write(f"MAE: {results['mae']:.6f}\n\n")
        
        f.write("Detailed results for each output dimension:\n")
        for i in range(len(results['r2_scores'])):
            f.write(f"Output {i+1}: R²={results['r2_scores'][i]:.6f}, "
                   f"MAPE={results['mape_scores'][i]:.6f}%, "
                   f"MAE={results['mae_scores'][i]:.6f}\n")
    
    print("\nTraining completed!")
    print(f"All results saved to folder: {experiment_folder}")
    
    # Restore standard output
    sys.stdout = original_stdout
    
    return results, training_time

if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparameters from config
    config = CONFIG
    
    print("Configuration parameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load data
    print("\nLoading data...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    adj_path = os.path.join(base_dir, config['adjacency_matrix_path'])
    train_data_path = os.path.join(base_dir, config['train_data_path'])
    
    edge_index, edge_weight = load_adjacency_matrix(
        adj_path, 
        gso_type=config['gso_type'],
        use_chebynet=config['use_chebynet'],
        K=config['chebynet_K']
    )
    pressure_data, head_loss_data = load_training_data(
        train_data_path, 
        pressure_cols=config['pressure_cols'],
        head_loss_cols=config['head_loss_cols']
    )
    
    print(f"Adjacency matrix shape: {edge_index.shape}")
    print(f"Pressure data shape: {pressure_data.shape}")
    print(f"Head loss data shape: {head_loss_data.shape}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, scaler_head_loss = create_data_loaders(
        pressure_data, head_loss_data, edge_index, edge_weight,
        seq_length=config['seq_length'],
        target_step=config['target_step'],
        batch_size=config['batch_size'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio']
    )
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Move edge_index and edge_weight to device
    edge_index_device = edge_index.to(device)
    edge_weight_device = edge_weight.to(device)
    
    # Execute single training run
    results, training_time = single_training_run(
        config, device, edge_index_device, edge_weight_device,
        train_loader, val_loader, test_loader, scaler_head_loss, 1
    )
    
    # Output summary
    print(f"\nSingle training summary:")
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"WMAPE: {results['wmape']:.6f}%")
    print(f"Average MAPE: {results['avg_mape']:.6f}%")
    print(f"Average MAE: {results['avg_mae']:.6f}")
    print(f"Average R²: {results['avg_r2']:.6f}")
    print(f"MSE: {results['mse']:.6f}")
    print(f"RMSE: {results['rmse']:.6f}")
    print(f"MAE: {results['mae']:.6f}")
    
    print("\nTraining completed!") 