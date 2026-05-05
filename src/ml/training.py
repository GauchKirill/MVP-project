import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

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
            min_delta: float = 1e-6,
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
        
        # Главный прогресс-бар по эпохам
        epoch_pbar = tqdm(range(epochs), desc="Обучение", unit="эпоха", ncols=120)
        
        for epoch in epoch_pbar:
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
            
            # Улучшение считается, если val_loss уменьшился хотя бы на min_delta
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Обновляем прогресс-бар
            epoch_pbar.set_postfix({
                'train': f'{train_loss:.4e}',
                'val': f'{val_loss:.4e}',
                'best': f'{best_val_loss:.4e}',
                'cap': f'{train_components.get("capacity", 0):.4e}',
                'dem': f'{train_components.get("demand", 0):.4e}',
                'pat': f'{patience_counter}/{early_stopping_patience}'
            })
            
            if patience_counter >= early_stopping_patience:
                epoch_pbar.write(f"\nРанняя остановка на эпохе {epoch}: "
                               f"val_loss не улучшался {early_stopping_patience} эпох "
                               f"(лучший val_loss = {best_val_loss:.6e})")
                break
        
        epoch_pbar.close()
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.history

    def _run_epoch(self, loader: DataLoader, capacity_masks: np.ndarray, 
                   training: bool = True):
        total_loss = 0.0
        total_components = {}
        num_batches = 0
        
        # Прогресс-бар для батчей
        desc = "Train" if training else "Val"
        batch_pbar = tqdm(loader, desc=desc, leave=False, unit="batch", ncols=100)
        
        for batch_idx, (batch_features, batch_demands) in enumerate(batch_pbar):
            batch_features = batch_features.to(self.device)
            batch_demands = batch_demands.to(self.device)
            
            # Извлекаем capacity из признаков (первые E элементов)
            batch_capacities = batch_features[:, :self.feature_extractor.E]
            
            # Индексы для маски
            batch_size_actual = batch_features.shape[0]
            start_idx = batch_idx * loader.batch_size if loader.batch_size else 0
            end_idx = min(start_idx + batch_size_actual, len(capacity_masks))
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
            
            # Обновляем прогресс-бар батчей
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4e}',
                'cap': f'{components.get("capacity", 0):.4e}',
                'dem': f'{components.get("demand", 0):.4e}',
                'over': f'{components.get("overloaded", 0):.0f}'
            })
        
        batch_pbar.close()
        
        return total_loss / num_batches, {k: v / num_batches for k, v in total_components.items()}

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float,
               components: Dict[str, float]):
        """Логирует информацию об эпохе."""
        comp_str = ", ".join([
            f"cap: {components.get('capacity', 0):.4e}",
            f"dem: {components.get('demand', 0):.4e}",
            f"avg_u: {components.get('avg_util', 0):.4e}",
            f"max_u: {components.get('max_util', 0):.4e}",
            f"deliv: {components.get('delivery_ratio', 0):.4e}",
            f"over: {components.get('overloaded', 0):.0f}"
        ])
        print(f"Epoch {epoch:3d} | Train: {train_loss:.4e} | Val: {val_loss:.4e}")
        print(f"         | {comp_str}")
        
        # ЗАПИСЬ ДЛЯ STREAMLIT — абсолютный путь
        try:
            epoch_file = os.path.join(os.getcwd(), "genereted", ".epoch")
            os.makedirs(os.path.dirname(epoch_file), exist_ok=True)
            with open(epoch_file, "w") as f:
                f.write(f"{epoch}\n{train_loss:.4e}\n{val_loss:.4e}")
        except Exception as e:
            print(f"DEBUG ERROR writing .epoch: {e}")

    def plot_loss_components(self, filename="loss_components.png"):
        """Отдельный график с компонентами train_loss"""
        if not self.history['train_components']:
            print("Нет данных для визуализации")
            return
        
        # Собираем только capacity и demand
        components = {}
        for comp_dict in self.history['train_components']:
            for key, value in comp_dict.items():
                if key in ['capacity', 'demand']:  # только эти два
                    if key not in components:
                        components[key] = []
                    components[key].append(value)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epochs = range(len(self.history['train_loss']))
        colors = {'capacity': '#e74c3c', 'demand': '#3498db'}
        
        for key, values in components.items():
            ax.plot(epochs, values, label=key, linewidth=1.5, color=colors.get(key, '#2ecc71'))
        
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Значение компонента')
        ax.set_title('Компоненты функции потерь (Train)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_dir = os.path.dirname(filename) if os.path.dirname(filename) else '.'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(filename, dpi=150)
        plt.close()  # не показывать, только сохранить
        print(f"График компонент сохранён в {filename}")

    def plot_loss_curves(self, filename="loss_curves.png"):
        """График train_loss и val_loss в логарифмическом масштабе"""
        if not self.history['train_loss']:
            print("Нет данных для визуализации")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epochs = range(len(self.history['train_loss']))
        
        ax.semilogy(epochs, self.history['train_loss'], 'b-', linewidth=1.5, label='Train Loss')
        if self.history['val_loss']:
            ax.semilogy(epochs, self.history['val_loss'], 'r-', linewidth=1.5, label='Validation Loss')
            
            # Отмечаем минимум валидационного лосса
            min_val_epoch = np.argmin(self.history['val_loss'])
            min_val_loss = self.history['val_loss'][min_val_epoch]
            ax.plot(min_val_epoch, min_val_loss, 'r*', markersize=10, 
                   label=f'Мин. val: {min_val_loss:.4e}')
        
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Функция потерь (логарифмический масштаб)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_dir = os.path.dirname(filename) if os.path.dirname(filename) else '.'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(filename, dpi=150)
        plt.close()  # не показывать, только сохранить
        print(f"График кривых обучения сохранён в {filename}")