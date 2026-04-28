import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class PathWeightNetwork(nn.Module):
    """
    Нейросеть для предсказания весов распределения потоков по путям.
    
    Вход: вектор признаков размерности E + S*C
    Выход: тензор размера (S, C, max_paths) с весами путей
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
            output_shape: (S, C, max_paths) - форма выходного тензора
            hidden_dims: размерности скрытых слоёв
            dropout_rate: вероятность dropout
            use_batch_norm: использовать ли BatchNorm
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_shape = output_shape
        self.S, self.C, self.max_paths = output_shape
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
        
        # Маска для несуществующих путей
        self.register_buffer('path_mask', torch.ones(self.output_shape))
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Используем Kaiming инициализацию для ReLU
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    # Инициализируем bias небольшими положительными значениями
                    nn.init.constant_(module.bias, 0.01)
    
    def set_path_mask(self, mask: np.ndarray):
        """
        Устанавливает маску для несуществующих путей.
        
        Args:
            mask: numpy массив размера (S, C, max_paths) с 1 для существующих путей
        """
        self.path_mask = torch.tensor(mask, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход.
        
        Args:
            x: входной тензор размера (batch_size, input_dim)
            
        Returns:
            тензор размера (batch_size, S, C, max_paths) с весами путей
        """
        batch_size = x.shape[0]
        
        # Пропускаем через энкодер
        features = self.encoder(x)
        
        # Получаем логиты
        logits = self.output_layer(features)
        
        # Переформатируем в (batch_size, S, C, max_paths)
        logits = logits.view(batch_size, self.S, self.C, self.max_paths)
        
        # Применяем маску (зануляем несуществующие пути перед softmax)
        masked_logits = logits.masked_fill(self.path_mask == 0, float('-inf'))
        
        # Softmax по последнему измерению (по путям)
        weights = F.softmax(masked_logits, dim=-1)
        
        # Для пар без путей заменяем NaN на 0
        weights = torch.nan_to_num(weights, nan=0.0)
        
        return weights
    
    def predict_flows(self, 
                    x: torch.Tensor, 
                    demands: torch.Tensor) -> torch.Tensor:
        """
        Предсказывает потоки на основе весов и заявок.
        
        Args:
            x: входной тензор признаков
            demands: тензор заявок размера (batch_size, S, C)
            
        Returns:
            тензор размера (batch_size, S, C, max_paths) с величинами потоков
        """
        weights = self.forward(x)
        # Поэлементное умножение: flow = weight * demand
        flows = weights * demands.unsqueeze(-1)
        return flows
