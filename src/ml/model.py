import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class PathWeightNetwork(nn.Module):
    """
    Нейронная модель для предсказания весов распределения потоков по путям
    
    Вход: вектор признаков размерности E + S*C
    Выход: тензор размера (S, C, max_paths + 1) с весами путей (включая фиктивный путь потерь)
    """
    def __init__(self, 
                input_dim: int,
                output_shape: Tuple[int, int, int],
                hidden_dims: Tuple[int, ...] = (512, 256, 128),
                dropout_rate: float = 0.3,
                use_batch_norm: bool = True):
        """
        Args:
            input_dim: размерность входного вектора признаков
            output_shape: (S, C, max_paths) - форма выходного тензора (без фиктивного пути)
            hidden_dims: размерности скрытых слоёв
            dropout_rate: вероятность dropout
            use_batch_norm: использовать ли BatchNorm
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_shape = output_shape
        self.S, self.C, self.max_real_paths = output_shape
        
        # Добавляем 1 к max_paths для фиктивного пути потерь
        self.max_paths = self.max_real_paths + 1
        self.output_flat_dim = self.S * self.C * self.max_paths
        
        # Строим энкодер
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Выходной слой
        self.output_layer = nn.Linear(prev_dim, self.output_flat_dim)
        
        # Маска для несуществующих путей (размер с учётом фиктивного пути)
        self.register_buffer('path_mask', torch.ones(self.S, self.C, self.max_paths))
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def set_path_mask(self, mask: np.ndarray):
        """
        Устанавливает маску для существующих путей.
        Фиктивный путь всегда разрешён (последний индекс).
        
        Args:
            mask: numpy массив размера (S, C, max_real_paths) с 1 для существующих путей
        """
        full_mask = np.ones((self.S, self.C, self.max_paths), dtype=np.float32)
        full_mask[:, :, :self.max_real_paths] = mask
        # Последний путь (фиктивный) всегда 1
        full_mask[:, :, -1] = 1.0
        self.path_mask = torch.tensor(full_mask, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: входной тензор размера (batch_size, input_dim)
            
        Returns:
            тензор размера (batch_size, S, C, max_paths) с весами путей
            (последний путь — фиктивный путь потерь)
        """
        batch_size = x.shape[0]
        
        # Пропускаем через энкодер
        features = self.encoder(x)
        
        # Получаем логиты
        logits = self.output_layer(features)
        
        # Переформатируем в (batch_size, S, C, max_paths)
        logits = logits.view(batch_size, self.S, self.C, self.max_paths)
        
        # Применяем мачку (зануляем несуществующие пути перед softmax)
        masked_logits = logits.masked_fill(self.path_mask == 0, float('-inf'))
        
        # Softmax по последнему измерению (по путям, включая фиктивный)
        weights = F.softmax(masked_logits, dim=-1)
        
        # Для пар без путей заменяем NaN на 0
        weights = torch.nan_to_num(weights, nan=0.0)
        
        return weights
    
    def predict_flows(self, 
                    x: torch.Tensor, 
                    demands: torch.Tensor) -> torch.Tensor:
        """
        Предсказывает потоки на основе весов и заявок.
        Фиктивный путь (последний) исключается из реальных потоков.
        
        Args:
            x: входной тензор признаков
            demands: тензор заявок размера (batch_size, S, C)
            
        Returns:
            тензор размера (batch_size, S, C, max_real_paths) с величинами потоков
            (без фиктивного пути)
        """
        weights = self.forward(x)  # (batch_size, S, C, max_paths)
        
        # Отделяем реальные пути от фиктивного
        real_weights = weights[:, :, :, :self.max_real_paths]  # (batch_size, S, C, max_real_paths)
        
        # Поэлементное умножение: flow = weight * demand
        flows = real_weights * demands.unsqueeze(-1)
        return flows
    
    def get_loss_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Возвращает веса фиктивного пути (доля потерь).
        """
        weights = self.forward(x)
        return weights[:, :, :, -1]  # (batch_size, S, C)