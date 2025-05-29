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
TRAIN_VAL_SAMPLES = 3  # 학습+검증 데이터 샘플 수
TEST_SAMPLES = 5000000        # 테스트 데이터 샘플 수
NUM_EPOCHS = 30           # 학습 에포크 수
BATCH_SIZE = 64           # 배치 크기
LEARNING_RATE = 0.001     # 학습률

# 로깅 설정
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
        
        # 가중치 초기화
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
        """
        Forward pass
        Args:
            x (torch.Tensor): 입력 텐서 [batch_size, input_size]
        Returns:
            torch.Tensor: 예측값 [batch_size, 1]
        """
        # 입력 reshape: [batch_size, 1, input_size]
        x = x.unsqueeze(1)
        
        # CNN 레이어 통과
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # 선형 레이어 통과
        x = self.fc_layers(x)
        
        return x
        
    def get_parameter_count(self):
        """모델의 총 파라미터 수를 반환"""
        return sum(p.numel() for p in self.parameters())

def load_and_preprocess_data():
    csv_files = glob.glob('../../logs/spike_features/*.csv')
    
    train_val_dfs = []
    test_dfs = []
    train_val_indices = []
    test_indices = []  # 테스트 데이터 파일 인덱스 추가
    
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
        
        # 학습+검증 데이터 샘플링
        if len(df) >= TRAIN_VAL_PER_FILE:
            train_val_df = df[:TRAIN_VAL_PER_FILE]
            train_val_indices.extend([file_idx] * TRAIN_VAL_PER_FILE)
        else:
            logger.warning(f"File {file} has less than {TRAIN_VAL_PER_FILE} rows for train+val: {len(df)}")
            train_val_df = df
            train_val_indices.extend([file_idx] * len(df))
        
        # 테스트 데이터 추출
        remaining_df = df[TRAIN_VAL_PER_FILE:]
        if len(remaining_df) >= TEST_PER_FILE:
            test_df = remaining_df[:TEST_PER_FILE]
            test_indices.extend([file_idx] * TEST_PER_FILE)  # 테스트 인덱스 추가
        else:
            logger.warning(f"File {file} has less than {TEST_PER_FILE} rows for test: {len(remaining_df)}")
            test_df = remaining_df
            test_indices.extend([file_idx] * len(test_df))  # 테스트 인덱스 추가
        
        train_val_dfs.append(train_val_df)
        test_dfs.append(test_df)
        
        logger.info(f"Added {len(train_val_df)} train+val samples and {len(test_df)} test samples from {file}")
    
    # 데이터프레임 합치기
    train_val_data = pd.concat(train_val_dfs, ignore_index=True)
    test_data = pd.concat(test_dfs, ignore_index=True)
    train_val_indices = np.array(train_val_indices)
    test_indices = np.array(test_indices)  # numpy 배열로 변환
    
    logger.info(f"\nFinal data sizes:")
    logger.info(f"Train+Validation set: {len(train_val_data)} rows")
    logger.info(f"Test set: {len(test_data)} rows")
    
    # 특성과 타겟 분리
    feature_columns = ['prev_spike_rate', 'mean_voltage', 'voltage_threshold_ratio', 
                      'cumulative_spikes', 'predicted_update_rate', 'train_loss']
    
    X_train_val = train_val_data[feature_columns]
    y_train_val = train_val_data['current_spike']
    
    X_test = test_data[feature_columns]
    y_test = test_data['current_spike']
    
    # 데이터 스케일링
    logger.info("\nScaling features...")
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 학습/검증 분할
    X_train, X_val, y_train, y_val, train_indices, val_indices = train_test_split(
        X_train_val_scaled, y_train_val.values, train_val_indices, 
        test_size=0.2, random_state=42
    )
    
    return (X_train, y_train, train_indices), (X_val, y_val, val_indices), \
           (X_test_scaled, y_test.values, test_indices), scaler, csv_files

def calculate_accuracy(predictions, actuals, tolerance=0.1):
    """
    주어진 허용 오차(tolerance) 범위 내의 예측 비율을 계산
    tolerance: 허용 오차 범위 (예: 0.1 = 10%)
    """
    # 실제값의 tolerance 범위 내에 예측값이 있는지 확인
    within_tolerance = np.abs(predictions - actuals) <= (np.abs(actuals) * tolerance)
    accuracy = np.mean(within_tolerance) * 100  # 백분율로 변환
    return accuracy

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in data_loader:
            # FileTrackingDataset의 경우 3개 값을 반환하므로 적절히 처리
            if len(batch) == 3:
                X_batch, y_batch, _ = batch  # 파일 인덱스는 무시
            else:
                X_batch, y_batch = batch
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            # 0차원 텐서 처리
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
    # 메트릭 계산
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
    
    """학습 곡선을 저장하는 함수"""
    try:
        
        import matplotlib.pyplot as plt
        
        
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        # 학습 손실 그래프
        plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        # 그래프 스타일링
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Training and Validation Loss\n{experiment_id}', fontsize=14, pad=20)
        plt.legend(fontsize=10)
        
        # y축 범위 설정 (이상치 제외)
        all_losses = train_losses + val_losses
        loss_mean = np.mean(all_losses)
        loss_std = np.std(all_losses)
        plt.ylim([max(0, loss_mean - 3*loss_std), loss_mean + 3*loss_std])
        
        # 파일 저장
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
    """평가 지표를 표 형식으로 변환"""
    table_data = []
    
    # 기본 메트릭 처리
    basic_metrics = {k: v for k, v in metrics_dict.items() 
                    if not isinstance(v, dict) and k not in ['Predictions Stats', 'Actuals Stats']}
    for metric_name, value in basic_metrics.items():
        table_data.append([metric_name, f"{value:.6f}"])
    
    # 통계 데이터 처리
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
        
        # 현재 파일의 데이터 인덱스 찾기
        file_mask = (self.file_indices == self.current_file_idx)
        file_data_indices = np.where(file_mask)[0]
        
        if self.current_sample_idx >= len(file_data_indices):
            self.current_file_idx += 1
            self.current_sample_idx = 0
            return self.__next__()
        
        # 현재 파일에서 배치 구성
        end_idx = min(self.current_sample_idx + self.batch_size, len(file_data_indices))
        batch_indices = file_data_indices[self.current_sample_idx:end_idx]
        
        # 배치 데이터 가져오기
        X_batch = torch.stack([self.dataset[i][0] for i in batch_indices])
        y_batch = torch.stack([self.dataset[i][1] for i in batch_indices])
        file_idx_batch = torch.tensor([self.dataset[i][2] for i in batch_indices])
        
        self.current_sample_idx = end_idx
        
        return X_batch, y_batch, file_idx_batch
    
    def __len__(self):
        return len(self.dataset) // self.batch_size + 1

def train_spike_predictor(batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE):
    # 타임스탬프를 포함한 실험 ID 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_id = f'e{epochs}_b{batch_size}_10000perfile_{timestamp}'
    
    # 로그 디렉토리 설정
    log_dir = f'log/predictor/{experiment_id}'
    os.makedirs(log_dir, exist_ok=True)
    
    # 로거 설정
    logger = setup_logger(log_dir)
    logger.info(f"Starting training with {epochs} epochs, {batch_size} batch size")
    
    # 모델 파일 경로
    model_path = os.path.join(log_dir, 'spike_predictor_cnn.pth')
    
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 데이터 로드 및 전처리
    logger.info("Loading and preprocessing data...")
    (X_train, y_train, train_indices), (X_val, y_val, val_indices), \
    (X_test, y_test, test_indices), scaler, csv_files = load_and_preprocess_data()
    
    # 데이터셋 생성 - test_dataset도 FileTrackingDataset 사용
    train_dataset = FileTrackingDataset(X_train, y_train, train_indices)
    val_dataset = FileTrackingDataset(X_val, y_val, val_indices)
    test_dataset = FileTrackingDataset(X_test, y_test, test_indices)  # SpikeDataset -> FileTrackingDataset
    
    # 데이터로더 생성
    train_loader = SequentialFileDataLoader(train_dataset, batch_size, train_indices, csv_files)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 모델 초기화
    model = SpikeUpdatePredictor(input_size=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 진행 상황 저장
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
            
            # 새로운 파일 시작 확인
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
            
            # 진행 바 업데이트
            train_pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'current_file': os.path.basename(csv_files[current_file]),
                'samples': f'{train_loader.current_sample_idx}/1000'
            })
        
        # 검증
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        with torch.no_grad():
            for X_batch, y_batch, _ in val_pbar:  # 파일 인덱스는 언더스코어로 무시
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # 평균 손실 계산
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 로깅
        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'  Train Loss: {avg_train_loss:.6f}')
        logger.info(f'  Val Loss: {avg_val_loss:.6f}')
        
        # 모델 저장
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
    
    # 테스트 세트에서 평가
    logger.info("Evaluating on test set...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 각 데이터셋에 대해 평가 수행
    train_metrics = evaluate_model(model, train_loader, device)
    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)
    
    # 최종 평가 결과 로깅
    logger.info("\nFinal Evaluation Results:")
    for dataset_name, metrics in [('Train', train_metrics), 
                                ('Validation', val_metrics), 
                                ('Test', test_metrics)]:
        logger.info(f"\n{dataset_name} Set Metrics:")
        logger.info("\n" + format_metrics_table(metrics))
    
    # 학습 곡선 저장
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
    """
    스파이크 예측 CNN 모델과 스케일러를 로드하는 함수
    
    Args:
        model_path (str): 체크포인트 파일 경로
        
    Returns:
        tuple: (model, scaler) - 로드된 모델과 스케일러
    """
    try:
        # GPU 사용 가능 여부 확인
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=device)
        
        # input_size가 없는 경우 기본값 설정
        input_size = checkpoint.get('input_size', 6)  # 기본 특성 수는 6
        
        # 체크포인트 구조 로깅
        logging.info("Checkpoint contents:")
        for key in checkpoint.keys():
            logging.info(f"  - {key}")
        
        # 모델 초기화 및 가중치 로드
        model = SpikeUpdatePredictor(input_size=input_size).to(device)
        
        # state_dict 키 확인
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 체크포인트가 직접 state dict인 경우
            model.load_state_dict(checkpoint)
            
        model.eval()  # 평가 모드로 설정
        
        # scaler 확인
        scaler = checkpoint.get('scaler', None)
        if scaler is None:
            logging.warning("Scaler not found in checkpoint. Using default StandardScaler.")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        
        # 로깅
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
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 특성 스케일링
    scaled_features = scaler.transform(features)
    
    # 텐서로 변환 및 GPU로 이동
    X = torch.FloatTensor(scaled_features).to(device)
    
    # 예측
    model.eval()
    with torch.no_grad():
        prediction = model(X)
    
    # CPU로 이동하여 numpy 배열로 변환
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
    parser = argparse.ArgumentParser(description='스파이크 업데이트 예측기 학습')
    parser.add_argument('--train', action='store_true', help='모델 학습 실행')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='에포크 수')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='학습률')
    
    args = parser.parse_args()
    
    if args.train:
        model, scaler, history = train_spike_predictor(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        # 실험 결과를 표 형식으로 출력
        print(f"\n실험 ID: {history['experiment_id']}")
        print("\n최종 평가 결과:")
        
        for dataset in ['train', 'val', 'test']:
            print(f"\n{dataset.upper()} 데이터셋:")
            metrics_table = format_metrics_table(history['final_metrics'][dataset])
            print("\n" + metrics_table)
    else:
        print("학습을 시작하려면 --train 플래그를 사용하세요.")
        print("예시: python spike_update_predictor.py --train")
    