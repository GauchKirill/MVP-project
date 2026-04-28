"""Извлечение признаков для нейросетевой модели."""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import Graph, RequestRegistry, Request


class FeatureExtractor:
    """
    Преобразует состояние сети и заявки в вектор признаков фиксированной размерности.
    
    Размерность вектора: E + S*C, где:
    - E: количество рёбер в графе
    - S: количество источников
    - C: количество потребителей
    
    Признаки:
    - первые E элементов: capacity рёбер (float('inf') для неограниченных)
    - остальные S*C: demands от источника к потребителю
    
    Нормализация через normalize_features():
    1. Глобальный максимум по всем конечным элементам
    2. Конечные значения делятся на этот максимум, inf остаются inf
    3. inf заменяются на 1.0
    """
    
    def __init__(self, graph: Graph, registry: RequestRegistry):
        self.graph = graph
        self.registry = registry
        
        # Фиксируем порядок источников, потребителей и рёбер
        self.sources = sorted(graph.get_sources(), key=lambda n: n.name)
        self.consumers = sorted(graph.get_consumers(), key=lambda n: n.name)
        self.edges = list(graph.edges)
        
        # Маппинги для быстрого доступа
        self.source_to_idx = {s.name: i for i, s in enumerate(self.sources)}
        self.consumer_to_idx = {c.name: i for i, c in enumerate(self.consumers)}
        
        # Размерности
        self.E = len(self.edges)
        self.S = len(self.sources)
        self.C = len(self.consumers)
        self.feature_dim = self.E + self.S * self.C
        
        # Максимальное число путей
        self.max_paths = self._find_max_paths()
        
    def _find_max_paths(self) -> int:
        """Находит максимальное количество путей среди всех заявок."""
        max_paths = 0
        for request in self.registry.requests:
            if request.paths:
                max_paths = max(max_paths, len(request.paths))
        return max_paths
    
    def build_raw_features(self, flows: Dict[str, Dict[str, float]]) -> np.ndarray:
        """
        Строит сырой вектор признаков (без нормализации) из словаря заявок.
        
        Args:
            flows: словарь вида {source: {consumer: demand}}
            
        Returns:
            numpy массив размера (feature_dim,) с сырыми значениями
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # 1. Capacity рёбер
        for i, edge in enumerate(self.edges):
            features[i] = edge.capacity  # float('inf') для неограниченных
        
        # 2. Заявки
        offset = self.E
        for s_name, consumers in flows.items():
            if s_name not in self.source_to_idx:
                continue
            s_idx = self.source_to_idx[s_name]
            for c_name, demand in consumers.items():
                if c_name not in self.consumer_to_idx:
                    continue
                c_idx = self.consumer_to_idx[c_name]
                features[offset + s_idx * self.C + c_idx] = demand
        
        return features
    
    def normalize_features(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Нормализует признаки:
        1. Находит глобальный максимум по всем конечным элементам (capacity + demands).
        2. Конечные значения делятся на этот максимум, inf остаются inf.
        3. inf заменяются на 1.0.
        
        Args:
            features: (feature_dim,) или (batch_size, feature_dim)
            
        Returns:
            normalized_features: той же формы
            capacity_mask: (E,) или (batch_size, E) — 1.0 для изначально конечных capacity,
                           0.0 для тех, что были inf
        """
        if features.ndim == 1:
            features = features.copy()
            # Маска конечных capacity до замен
            capacity_mask = np.isfinite(features[:self.E]).astype(np.float32)
            
            # Глобальный максимум только по конечным значениям
            finite_vals = features[np.isfinite(features)]
            global_max = finite_vals.max() if len(finite_vals) > 0 else 1.0
            
            # Деление конечных, inf остаются inf
            features = np.where(np.isfinite(features), features / global_max, features)
            # Замена inf → 1.0
            features[~np.isfinite(features)] = 1.0
            
            return features, capacity_mask
        
        else:
            batch_size = features.shape[0]
            features = features.copy()
            capacity_mask = np.zeros((batch_size, self.E), dtype=np.float32)
            
            for i in range(batch_size):
                row = features[i]
                cap_mask = np.isfinite(row[:self.E])
                capacity_mask[i] = cap_mask.astype(np.float32)
                
                finite_vals = row[np.isfinite(row)]
                global_max = finite_vals.max() if len(finite_vals) > 0 else 1.0
                
                row = np.where(np.isfinite(row), row / global_max, row)
                row[~np.isfinite(row)] = 1.0
                features[i] = row
            
            return features, capacity_mask
    
    def get_output_shape(self) -> Tuple[int, int, int]:
        """Возвращает форму выходного тензора: (S, C, max_paths)."""
        return (self.S, self.C, self.max_paths)
    
    def create_path_mask(self) -> np.ndarray:
        """Создаёт маску для выходного слоя: 1 для существующих путей, 0 для несуществующих."""
        mask = np.zeros((self.S, self.C, self.max_paths), dtype=np.float32)
        
        for request in self.registry.requests:
            s_name = request.source.name
            c_name = request.consumer.name
            
            if s_name in self.source_to_idx and c_name in self.consumer_to_idx:
                s_idx = self.source_to_idx[s_name]
                c_idx = self.consumer_to_idx[c_name]
                num_paths = len(request.paths)
                
                if num_paths > 0:
                    mask[s_idx, c_idx, :num_paths] = 1.0
        
        return mask
    
    def get_edge_capacities(self) -> np.ndarray:
        """
        Возвращает массив пропускных способностей рёбер.
        Используется в loss-функции для вычисления превышений.
        """
        caps = np.zeros(self.E, dtype=np.float32)
        for i, edge in enumerate(self.edges):
            caps[i] = edge.capacity
        return caps