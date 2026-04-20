"""Извлечение признаков для нейросетевой модели."""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# Добавляем родительскую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import Graph, RequestRegistry, Request


class FeatureExtractor:
    """
    Преобразует состояние сети и заявки в вектор признаков фиксированной размерности.
    
    Размерность вектора: E + S*C, где:
    - E: количество рёбер в графе
    - S: количество источников
    - C: количество потребителей
    
    Признаки нормализуются делением на сумму всех значений.
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
        
        # Максимальное число путей (определяется эмпирически)
        self.max_paths = self._find_max_paths()
        
    def _find_max_paths(self) -> int:
        """Находит максимальное количество путей среди всех заявок."""
        max_paths = 0
        for request in self.registry.requests:
            if request.paths:
                max_paths = max(max_paths, len(request.paths))
        return max_paths
    
    def extract_features(self, 
                         flows: Dict[str, Dict[str, float]], 
                         normalize: bool = True) -> np.ndarray:
        """
        Извлекает вектор признаков из данных о потоках.
        
        Args:
            flows: словарь вида {source: {consumer: demand}}
            normalize: нормализовать ли признаки делением на сумму
            
        Returns:
            numpy массив размера (feature_dim,)
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # 1. Заполняем capacity рёбер (первые E признаков)
        for i, edge in enumerate(self.edges):
            if edge.capacity == float('inf'):
                features[i] = 1.0  # inf → 1 (максимальная возможная доля)
            else:
                features[i] = edge.capacity
        
        # 2. Заполняем заявки (оставшиеся S*C признаков)
        offset = self.E
        for s_name, consumers in flows.items():
            if s_name not in self.source_to_idx:
                continue
            s_idx = self.source_to_idx[s_name]
            for c_name, demand in consumers.items():
                if c_name not in self.consumer_to_idx:
                    continue
                c_idx = self.consumer_to_idx[c_name]
                flat_idx = offset + s_idx * self.C + c_idx
                features[flat_idx] = demand
        
        # 3. Нормализация - ВСЁ делим на сумму
        if normalize and features.sum() > 0:
            features = features / features.sum()
        
        return features
    
    def extract_batch_features(self, 
                               flows_list: List[Dict], 
                               normalize: bool = True) -> np.ndarray:
        """
        Извлекает признаки для батча сценариев.
        
        Returns:
            numpy массив размера (batch_size, feature_dim)
        """
        batch_features = np.zeros((len(flows_list), self.feature_dim), dtype=np.float32)
        for i, flows in enumerate(flows_list):
            batch_features[i] = self.extract_features(flows, normalize=False)
        
        # Нормализуем каждый сценарий отдельно
        if normalize:
            row_sums = batch_features.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0  # избегаем деления на 0
            batch_features = batch_features / row_sums
        
        return batch_features
    
    def get_output_shape(self) -> Tuple[int, int, int]:
        """
        Возвращает форму выходного тензора.
        
        Returns:
            (S, C, max_paths) - трёхмерная матрица весов путей
        """
        return (self.S, self.C, self.max_paths)
    
    def create_path_mask(self) -> np.ndarray:
        """
        Создаёт маску для выходного слоя.
        
        Маска содержит 1 для существующих путей и 0 для несуществующих.
        Используется для маскирования логитов перед softmax.
        
        Returns:
            numpy массив размера (S, C, max_paths)
        """
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
        Возвращает массив пропускных способностей всех рёбер.
        
        Returns:
            numpy массив размера (E,)
        """
        caps = np.zeros(self.E, dtype=np.float32)
        for i, edge in enumerate(self.edges):
            caps[i] = edge.capacity if edge.capacity != float('inf') else 1e9
        return caps
    
    def get_edge_capacities_normalized(self, total_sum: float = 1.0) -> np.ndarray:
        """
        Возвращает нормализованные пропускные способности рёбер.
        Они соответствуют тому же масштабу, что и признаки.
        
        Args:
            total_sum: примерная сумма всех значений для нормализации
            
        Returns:
            numpy массив размера (E,) с нормализованными capacity
        """
        caps = np.zeros(self.E, dtype=np.float32)
        
        # Вычисляем общую сумму для нормализации
        # Используем сумму всех capacity + total_sum (для учёта demands)
        total_capacity_sum = 0.0
        for edge in self.edges:
            if edge.capacity != float('inf'):
                total_capacity_sum += edge.capacity
        
        normalizer = total_capacity_sum + total_sum
        
        for i, edge in enumerate(self.edges):
            if edge.capacity == float('inf'):
                caps[i] = 1.0  # inf → максимальная доля
            else:
                caps[i] = edge.capacity / normalizer
        
        return caps