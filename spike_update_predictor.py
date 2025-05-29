import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import os
from tqdm import tqdm
import logging
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tabulate import tabulate
import argparse

__all__ = ['SpikeUpdatePredictor', 'train_spike_predictor', 'load_predictor', 'predict_spike']

# 상수 정의
TRAIN_VAL_SAMPLES = 3  
TEST_SAMPLES = 5000000       
NUM_EPOCHS = 30           
BATCH_SIZE = 64           
LEARNING_RATE = 0.001     

def setup_logger(log_dir='log/predictor'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class SpikeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SpikeUpdatePredictor(nn.Module):
    def __init__(self, input_size=6):
        super(SpikeUpdatePredictor, self).__init__()
        
        # 1D CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """가중치 초기화 메소드"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x
        
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

def load_and_preprocess_data():
    csv_files = glob.glob('../../logs/spike_features/*.csv')
    
    train_val_dfs = []
    test_dfs = []
    train_val_indices = []
    test_indices = [] 
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data files...")
    
    TRAIN_VAL_PER_FILE = 10000
    TEST_PER_FILE = 100000
    
    for file_idx, file in enumerate(csv_files):
        logger.info(f"Processing file: {file}")
        df = pd.read_csv(file)
        initial_rows = len(df)
        
        df = df[df['current_spike'] >= 0.0]
        df = df.dropna()
        removed_rows = initial_rows - len(df)
        logger.info(f"Removed {removed_rows} rows containing null values from {file}")
        
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        
        if len(df) >= TRAIN_VAL_PER_FILE:
            train_val_df = df[:TRAIN_VAL_PER_FILE]
            train_val_indices.extend([file_idx] * TRAIN_VAL_PER_FILE)
        else:
            logger.warning(f"File {file} has less than {TRAIN_VAL_PER_FILE} rows for train+val: {len(df)}")
            train_val_df = df
            train_val_indices.extend([file_idx] * len(df))
        
        
        remaining_df = df[TRAIN_VAL_PER_FILE:]
        if len(remaining_df) >= TEST_PER_FILE:
            test_df = remaining_df[:TEST_PER_FILE]
            test_indices.extend([file_idx] * TEST_PER_FILE)  
        else:
            logger.warning(f"File {file} has less than {TEST_PER_FILE} rows for test: {len(remaining_df)}")
            test_df = remaining_df
            test_indices.extend([file_idx] * len(test_df)) 
        
        train_val_dfs.append(train_val_df)
        test_dfs.append(test_df)
        
        logger.info(f"Added {len(train_val_df)} train+val samples and {len(test_df)} test samples from {file}")
    

    train_val_data = pd.concat(train_val_dfs, ignore_index=True)
    test_data = pd.concat(test_dfs, ignore_index=True)
    train_val_indices = np.array(train_val_indices)
    test_indices = np.array(test_indices)  
    
    logger.info(f"\nFinal data sizes:")
    logger.info(f"Train+Validation set: {len(train_val_data)} rows")
    logger.info(f"Test set: {len(test_data)} rows")
    
   
    feature_columns = ['prev_spike_rate', 'mean_voltage', 'voltage_threshold_ratio', 
                      'cumulative_spikes', 'predicted_update_rate', 'train_loss']
    
    X_train_val = train_val_data[feature_columns]
    y_train_val = train_val_data['current_spike']
    
    X_test = test_data[feature_columns]
    y_test = test_data['current_spike']
    
    
    logger.info("\nScaling features...")
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)
    
    
    X_train, X_val, y_train, y_val, train_indices, val_indices = train_test_split(
        X_train_val_scaled, y_train_val.values, train_val_indices, 
        test_size=0.2, random_state=42
    )
    
    return (X_train, y_train, train_indices), (X_val, y_val, val_indices), \
           (X_test_scaled, y_test.values, test_indices), scaler, csv_files

def calculate_accuracy(predictions, actuals, tolerance=0.1):
    
    within_tolerance = np.abs(predictions - actuals) <= (np.abs(actuals) * tolerance)
    accuracy = np.mean(within_tolerance) * 100  
    return accuracy

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in data_loader:
            
            if len(batch) == 3:
                X_batch, y_batch, _ = batch 
            else:
                X_batch, y_batch = batch
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)

            if outputs.dim() == 0:
                predictions.append(outputs.item())
            else:
                predictions.extend(outputs.squeeze().cpu().numpy().flatten())
            
            if y_batch.dim() == 0:
                actuals.append(y_batch.item())
            else:
                actuals.extend(y_batch.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    

    accuracies = {
        'Accuracy (5%)': calculate_accuracy(predictions, actuals, 0.05),
        'Accuracy (10%)': calculate_accuracy(predictions, actuals, 0.10),
        'Accuracy (20%)': calculate_accuracy(predictions, actuals, 0.20)
    }
  
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    
    return {
        **accuracies,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def save_training_curve(train_losses, val_losses, experiment_id, log_dir, logger):
    
    try:
        
        import matplotlib.pyplot as plt
        
        
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Training and Validation Loss\n{experiment_id}', fontsize=14, pad=20)
        plt.legend(fontsize=10)
        
        all_losses = train_losses + val_losses
        loss_mean = np.mean(all_losses)
        loss_std = np.std(all_losses)
        plt.ylim([max(0, loss_mean - 3*loss_std), loss_mean + 3*loss_std])
        
    
        plot_path = os.path.join(log_dir, 'training_curve.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Successfully saved training curve to {plot_path}")
        
        
        
    except ImportError as e:
        logger.error(f"Required plotting libraries not installed: {e}")
    except Exception as e:
        logger.error(f"Error saving training curve: {str(e)}")
        logger.error(f"Train losses: {train_losses}")
        logger.error(f"Val losses: {val_losses}")

def format_metrics_table(metrics_dict):

    table_data = []
    
    basic_metrics = {k: v for k, v in metrics_dict.items() 
                    if not isinstance(v, dict) and k not in ['Predictions Stats', 'Actuals Stats']}
    for metric_name, value in basic_metrics.items():
        table_data.append([metric_name, f"{value:.6f}"])
    
    
    for stats_name in ['Predictions Stats', 'Actuals Stats']:
        if stats_name in metrics_dict:
            table_data.append([f"\n{stats_name}", ""])
            for k, v in metrics_dict[stats_name].items():
                table_data.append([f"  {k}", f"{v:.6f}"])
    
    return tabulate(table_data, headers=['Metric', 'Value'], 
                   tablefmt='grid', floatfmt='.6f')

class SequentialFileDataLoader:
    def __init__(self, dataset, batch_size, file_indices, csv_files):
        self.dataset = dataset
        self.batch_size = batch_size
        self.file_indices = file_indices
        self.csv_files = csv_files
        self.current_file_idx = 0
        self.current_sample_idx = 0
        
    def __iter__(self):
        self.current_file_idx = 0
        self.current_sample_idx = 0
        return self
    
    def __next__(self):
        if self.current_file_idx >= len(self.csv_files):
            raise StopIteration
        
        
        file_mask = (self.file_indices == self.current_file_idx)
        file_data_indices = np.where(file_mask)[0]
        
        if self.current_sample_idx >= len(file_data_indices):
            self.current_file_idx += 1
            self.current_sample_idx = 0
            return self.__next__()
        
        
        end_idx = min(self.current_sample_idx + self.batch_size, len(file_data_indices))
        batch_indices = file_data_indices[self.current_sample_idx:end_idx]
        
        
        X_batch = torch.stack([self.dataset[i][0] for i in batch_indices])
        y_batch = torch.stack([self.dataset[i][1] for i in batch_indices])
        file_idx_batch = torch.tensor([self.dataset[i][2] for i in batch_indices])
        
        self.current_sample_idx = end_idx
        
        return X_batch, y_batch, file_idx_batch
    
    def __len__(self):
        return len(self.dataset) // self.batch_size + 1

def train_spike_predictor(batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE):
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_id = f'e{epochs}_b{batch_size}_10000perfile_{timestamp}'
    
    
    log_dir = f'log/predictor/{experiment_id}'
    os.makedirs(log_dir, exist_ok=True)
    
    
    logger = setup_logger(log_dir)
    logger.info(f"Starting training with {epochs} epochs, {batch_size} batch size")
    
    
    model_path = os.path.join(log_dir, 'spike_predictor_cnn.pth')
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    

    logger.info("Loading and preprocessing data...")
    (X_train, y_train, train_indices), (X_val, y_val, val_indices), \
    (X_test, y_test, test_indices), scaler, csv_files = load_and_preprocess_data()
        
    train_dataset = FileTrackingDataset(X_train, y_train, train_indices)
    val_dataset = FileTrackingDataset(X_val, y_val, val_indices)
    test_dataset = FileTrackingDataset(X_test, y_test, test_indices) 
    

    train_loader = SequentialFileDataLoader(train_dataset, batch_size, train_indices, csv_files)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
  
    model = SpikeUpdatePredictor(input_size=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
   
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    logger.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        current_file = None
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for i, (X_batch, y_batch, file_idx_batch) in enumerate(train_pbar):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
          
            batch_file_idx = file_idx_batch[0].item()
            if current_file != batch_file_idx:
                current_file = batch_file_idx
                logger.info(f"\nEpoch {epoch+1}: Starting to process file: {csv_files[current_file]}")
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'current_file': os.path.basename(csv_files[current_file]),
                'samples': f'{train_loader.current_sample_idx}/1000'
            })
        
     
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        with torch.no_grad():
            for X_batch, y_batch, _ in val_pbar: 
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
     
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
      
        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'  Train Loss: {avg_train_loss:.6f}')
        logger.info(f'  Val Loss: {avg_val_loss:.6f}')
      
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'scaler': scaler,
                'input_size': X_train.shape[1],
                'experiment_id': experiment_id,
                'config': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'train_val_samples': len(X_train) + len(X_test),
                    'test_samples': len(X_test),
                    'learning_rate': learning_rate
                }
            }, model_path)
            logger.info(f'  Saved best model with validation loss: {avg_val_loss:.6f}')
    
 
    logger.info("Evaluating on test set...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
  
    train_metrics = evaluate_model(model, train_loader, device)
    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)

    logger.info("\nFinal Evaluation Results:")
    for dataset_name, metrics in [('Train', train_metrics), 
                                ('Validation', val_metrics), 
                                ('Test', test_metrics)]:
        logger.info(f"\n{dataset_name} Set Metrics:")
        logger.info("\n" + format_metrics_table(metrics))
    

    save_training_curve(
        train_losses=train_losses,
        val_losses=val_losses,
        experiment_id=experiment_id,
        log_dir=log_dir,
        logger=logger
    )
    
    return model, scaler, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        },
        'experiment_id': experiment_id,
        'log_dir': log_dir
    }

def load_predictor(model_path='spike_predictor_cnn.pth'):

    try:
       
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
     
        checkpoint = torch.load(model_path, map_location=device)
        
        input_size = checkpoint.get('input_size', 6)  
        
       
        logging.info("Checkpoint contents:")
        for key in checkpoint.keys():
            logging.info(f"  - {key}")
        

        model = SpikeUpdatePredictor(input_size=input_size).to(device)
        
       
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
           
            model.load_state_dict(checkpoint)
            
        model.eval() 
        
       
        scaler = checkpoint.get('scaler', None)
        if scaler is None:
            logging.warning("Scaler not found in checkpoint. Using default StandardScaler.")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        
       
        logging.info(f"Successfully loaded predictor from {model_path}")
        if 'config' in checkpoint:
            logging.info(f"Model configuration: {checkpoint['config']}")
        
        return model, scaler
        
    except FileNotFoundError:
        logging.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading predictor from {model_path}: {str(e)}")
        logging.error(f"Checkpoint structure: {checkpoint.keys() if 'checkpoint' in locals() else 'Not loaded'}")
        raise

def predict_spike(model, scaler, features):
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
   
    scaled_features = scaler.transform(features)
    
   
    X = torch.FloatTensor(scaled_features).to(device)
    
   
    model.eval()
    with torch.no_grad():
        prediction = model(X)
    
   
    return prediction.cpu().numpy()

class FileTrackingDataset(Dataset):
    def __init__(self, X, y, file_indices):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.file_indices = file_indices
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.file_indices[idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spike Update Predictor Training')
    parser.add_argument('--train', action='store_true', help='Run model training')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='Learning rate')
    
    args = parser.parse_args()
    
    if args.train:
        model, scaler, history = train_spike_predictor(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
       
        print(f"\nExperiment ID: {history['experiment_id']}")
        print("\nFinal Evaluation Results:")
        
        for dataset in ['train', 'val', 'test']:
            print(f"\n{dataset.upper()} Dataset:")
            metrics_table = format_metrics_table(history['final_metrics'][dataset])
            print("\n" + metrics_table)
    else:
        print("To start training, use the --train flag.")
        print("Example: python spike_update_predictor.py --train")
    