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
    Класс для обучения модели распределения потоков
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
        Настраивает оптимизатор и планировщик
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
            train_capacity_masks: np.ndarray,
            val_features: np.ndarray,
            val_demands: np.ndarray,
            val_capacity_masks: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            early_stopping_patience: int = 30,
            verbose: bool = True) -> Dict:
        
        train_dataset = TensorDataset(
            torch.FloatTensor(train_features),
            torch.FloatTensor(train_demands)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_features is not None and val_demands is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(val_features),
                torch.FloatTensor(val_demands)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss, train_components = self._run_epoch(
                train_loader, train_capacity_masks, training=True
            )
            
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss, val_components = self._run_epoch(
                        val_loader, val_capacity_masks, training=False
                    )
            else:
                val_loss = train_loss
                val_components = train_components
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_components'].append(train_components)
            self.history['val_components'].append(val_components)
            
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose:
                self._log_epoch(epoch, train_loss, val_loss, train_components)
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nРанняя остановка на эпохе {epoch}")
                break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.history


    def _run_epoch(self, loader: DataLoader, capacity_masks: np.ndarray, training: bool = True):
        total_loss = 0.0
        total_components = {}
        num_batches = 0
        
        # Оборачиваем loader в tqdm для отображения прогресса батчей
        # desc меняется в зависимости от режима (train/val)
        desc = "Training" if training else "Validation"
        progress_bar = tqdm(loader, desc=desc, leave=False, unit="batch")
        
        for batch_idx, (batch_features, batch_demands) in enumerate(progress_bar):
            batch_features = batch_features.to(self.device)
            batch_demands = batch_demands.to(self.device)
            
            # Извлекаем capacity из признаков (первые E элементов)
            batch_capacities = batch_features[:, :self.feature_extractor.E]
            
            # Индексы для маски
            batch_size = loader.batch_size if loader.batch_size is not None else 1
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(capacity_masks))
            batch_mask = torch.FloatTensor(capacity_masks[start_idx:end_idx]).to(self.device)
            
            if training and self.optimizer:
                self.optimizer.zero_grad()
            
            path_flows = self.model.predict_flows(batch_features, batch_demands)
            edge_flows = self.edge_calculator.compute_edge_flows(path_flows)
            
            loss, components = self.loss_fn(
                path_flows=path_flows,
                edge_flows=edge_flows,
                demands=batch_demands,
                edge_capacities=batch_capacities,
                capacity_mask=batch_mask
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
            
            # Обновляем описание прогресс-бара текущими метриками
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4e}',
                'cap': f'{components.get("capacity", 0):.4e}',
                'dem': f'{components.get("demand", 0):.4e}',
                'avg_u': f'{components.get("avg_util", 0):.3f}',
                'over': f'{components.get("overloaded", 0):.0f}'
            })
        
        return total_loss / num_batches, {k: v / num_batches for k, v in total_components.items()}

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                components: Dict[str, float]):
        print(
            f"Epoch {epoch:3d} | "
            f"Train: {train_loss:.4e} | "
            f"Val: {val_loss:.4e} | "
            f"cap: {components.get('capacity', 0):.4e} | "
            f"dem: {components.get('demand', 0):.4e} | "
            f"avg_u: {components.get('avg_util', 0):.4e} | "
            f"max_u: {components.get('max_util', 0):.4e} | "
            f"over: {components.get('overloaded', 0):.0f}"
        )