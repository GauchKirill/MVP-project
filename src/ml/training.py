"""Обучение нейросетевой модели."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

from .model import PathWeightNetwork
from .loss import PowerFlowLoss, EdgeFlowCalculator
from .feature_extractor import FeatureExtractor


class ModelTrainer:
    """
    Класс для обучения модели распределения потоков.
    """
    
    def __init__(self,
                 model: PathWeightNetwork,
                 feature_extractor: FeatureExtractor,
                 edge_calculator: EdgeFlowCalculator,
                 loss_fn: PowerFlowLoss,
                 device: str = 'cpu'):
        
        self.model = model.to(device)
        self.feature_extractor = feature_extractor
        self.edge_calculator = edge_calculator
        self.loss_fn = loss_fn
        self.device = device
        
        self.optimizer = None
        self.scheduler = None
        
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_components': [],
            'val_components': []
        }
    
    def configure_optimizer(self, 
                            lr: float = 1e-3,
                            weight_decay: float = 1e-5,
                            scheduler_patience: int = 20):
        """
        Настраивает оптимизатор и планировщик.
        """
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=scheduler_patience
        )
    
    def train(self,
              train_features: np.ndarray,
              train_demands: np.ndarray,
              val_features: Optional[np.ndarray] = None,
              val_demands: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              early_stopping_patience: int = 30,
              verbose: bool = True) -> Dict:
        """
        Обучает модель.
        
        Args:
            train_features: признаки обучающей выборки (N, feature_dim)
            train_demands: заявки обучающей выборки (N, S, C)
            val_features: признаки валидационной выборки
            val_demands: заявки валидационной выборки
            epochs: количество эпох
            batch_size: размер батча
            early_stopping_patience: терпение для ранней остановки
            verbose: печатать ли прогресс
            
        Returns:
            словарь с историей обучения
        """
        # Создаём даталоадеры
        train_dataset = TensorDataset(
            torch.FloatTensor(train_features),
            torch.FloatTensor(train_demands)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_features is not None and val_demands is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(val_features),
                torch.FloatTensor(val_demands)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Ранняя остановка
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Основной цикл обучения
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss, train_components = self._run_epoch(train_loader, training=True)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss, val_components = self._run_epoch(val_loader, training=False)
            else:
                val_loss = train_loss
                val_components = train_components
            
            # Сохраняем историю
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_components'].append(train_components)
            self.history['val_components'].append(val_components)
            
            # Обновляем learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Ранняя остановка
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Логирование
            if verbose and (epoch % 3 == 0 or epoch == epochs - 1):
                self._log_epoch(epoch, train_loss, val_loss, train_components)
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nРанняя остановка на эпохе {epoch}")
                break
        
        # Восстанавливаем лучшую модель
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.history
    
    def _run_epoch(self, loader: DataLoader, training: bool = True) -> Tuple[float, Dict[str, float]]:
        """
        Прогоняет одну эпоху.
        """
        total_loss = 0.0
        total_components = {}
        num_batches = 0
        
        # Получаем реальные capacity в кВт
        edge_capacities = torch.FloatTensor(
            self.feature_extractor.get_edge_capacities()
        ).to(self.device)
        
        for batch_features, batch_demands in loader:
            batch_features = batch_features.to(self.device)
            batch_demands = batch_demands.to(self.device)
            
            if training:
                if self.optimizer:
                    self.optimizer.zero_grad()
            
            # Прямой проход - получаем веса (коэффициенты от 0 до 1)
            path_weights = self.model(batch_features)
            
            # Потоки в кВт = веса * demands (demands уже в кВт)
            path_flows = path_weights * batch_demands.unsqueeze(-1)
            
            # Вычисляем потоки на рёбрах в кВт
            edge_flows = self.edge_calculator.compute_edge_flows(path_flows)
            
            # Вычисляем функцию потерь
            loss, components = self.loss_fn(
                path_flows=path_flows,
                edge_flows=edge_flows,
                demands=batch_demands,
                edge_capacities=edge_capacities
            )
            
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if self.optimizer:
                    self.optimizer.step()
            
            total_loss += loss.item()
            for key, value in components.items():
                total_components[key] = total_components.get(key, 0.0) + value
            
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                   components: Dict[str, float]):
        """Логирует информацию об эпохе."""
        comp_str = ", ".join([f"{k}: {v:.4f}" for k, v in components.items()])
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"           | {comp_str}")