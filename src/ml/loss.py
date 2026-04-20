import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class PowerFlowLoss(nn.Module):
    """
    Функция потерь для оптимизации потокораспределения.
    
    Компоненты:
    1. capacity_loss — штраф за превышение пропускной способности рёбер
    2. demand_loss — штраф за недопоставку энергии
    
    Правило Кирхгофа (сохранение потока в узлах) выполняется автоматически,
    так как потоки идут только по физически существующим путям.
    """
    
    def __init__(self,
        capacity_weight: float = 10.0,
        demand_weight: float = 100.0):
        """
        Args:
            capacity_weight: вес штрафа за превышение пропускной способности
            demand_weight: вес штрафа за недопоставку энергии
        """
        super().__init__()
        
        self.capacity_weight = capacity_weight
        self.demand_weight = demand_weight
        
        # Для логирования
        self.last_losses: Dict[str, float] = {}
    
    def forward(self,
                path_flows: torch.Tensor,           # (batch, S, C, max_paths)
                edge_flows: torch.Tensor,           # (batch, E)
                demands: torch.Tensor,              # (batch, S, C)
                edge_capacities: torch.Tensor,      # (E,)
                path_lengths: Optional[torch.Tensor] = None,  # не используется
                path_masks: Optional[torch.Tensor] = None     # не используется
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Вычисляет функцию потерь.
        
        Args:
            path_flows: потоки по каждому пути (batch, S, C, max_paths)
            edge_flows: суммарные потоки на рёбрах (batch, E)
            demands: заявки (batch, S, C)
            edge_capacities: пропускные способности рёбер (E,)
            path_lengths: игнорируется (оставлен для совместимости)
            path_masks: игнорируется (оставлен для совместимости)
            
        Returns:
            total_loss: скалярное значение функции потерь
            components: словарь с компонентами для логирования
        """
        components = {}
        total_loss = torch.tensor(0.0, device=path_flows.device)
        
        # 1. Capacity loss — штраф за превышение пропускной способности
        edge_capacities_expanded = edge_capacities.unsqueeze(0)  # (1, E)
        capacity_violation = F.relu(edge_flows - edge_capacities_expanded)
        loss_capacity = self.capacity_weight * (capacity_violation ** 2).mean()
        total_loss = total_loss + loss_capacity
        components['capacity'] = loss_capacity.item()
        
        # 2. Demand loss — штраф за недопоставку
        delivered = path_flows.sum(dim=-1)  # (batch, S, C)
        shortage = F.relu(demands - delivered)
        loss_demand = self.demand_weight * (shortage ** 2).mean()
        total_loss = total_loss + loss_demand
        components['demand'] = loss_demand.item()
        
        # Логируем дополнительные метрики (не влияют на градиент)
        with torch.no_grad():
            # Средняя загрузка рёбер
            edge_utils = edge_flows / edge_capacities_expanded.clamp(min=1.0)
            components['avg_utilization'] = edge_utils.mean().item()
            
            # Доля доставленной энергии
            total_demanded = demands.sum()
            total_delivered = delivered.sum()
            if total_demanded > 0:
                components['delivery_ratio'] = (total_delivered / total_demanded).item()
            else:
                components['delivery_ratio'] = 1.0
        
        self.last_losses = components
        return total_loss, components
    
    def get_last_losses(self) -> Dict[str, float]:
        """Возвращает компоненты функции потерь с последнего вызова."""
        return self.last_losses.copy()


class EdgeFlowCalculator:
    """
    Вычисляет суммарные потоки на рёбрах на основе потоков по путям.
    """
    
    def __init__(self, 
        registry,  # RequestRegistry
        feature_extractor):
        self.registry = registry
        self.extractor = feature_extractor
        
        # Строим маппинг: (s_idx, c_idx, path_idx) -> список индексов рёбер
        self.path_to_edges = self._build_path_to_edges_mapping()
        
    def _build_path_to_edges_mapping(self) -> Dict[Tuple[int, int, int], List[int]]:
        """
        Строит отображение из индекса пути в индексы рёбер.
        
        Returns:
            словарь {(s_idx, c_idx, path_idx): [edge_idx, ...]}
        """
        mapping = {}
        
        # Создаём маппинг рёбер для быстрого поиска
        edge_to_idx = {edge: i for i, edge in enumerate(self.extractor.edges)}
        
        for request in self.registry.requests:
            s_name = request.source.name
            c_name = request.consumer.name
            
            if s_name not in self.extractor.source_to_idx:
                continue
            if c_name not in self.extractor.consumer_to_idx:
                continue
            
            s_idx = self.extractor.source_to_idx[s_name]
            c_idx = self.extractor.consumer_to_idx[c_name]
            
            for path_idx, path in enumerate(request.paths):
                edge_indices = []
                for edge in path:
                    if edge in edge_to_idx:
                        edge_indices.append(edge_to_idx[edge])
                
                mapping[(s_idx, c_idx, path_idx)] = edge_indices
        
        return mapping
    
    def compute_edge_flows(self, 
        path_flows: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет суммарные потоки на рёбрах.
        
        Args:
            path_flows: тензор (batch_size, S, C, max_paths)
            
        Returns:
            тензор (batch_size, E) с суммарными потоками
        """
        batch_size = path_flows.shape[0]
        E = self.extractor.E
        device = path_flows.device
        
        edge_flows = torch.zeros(batch_size, E, device=device)
        
        for (s_idx, c_idx, path_idx), edge_indices in self.path_to_edges.items():
            # Берём поток по данному пути
            flow = path_flows[:, s_idx, c_idx, path_idx]  # (batch_size,)
            
            # Добавляем к каждому ребру пути
            for edge_idx in edge_indices:
                edge_flows[:, edge_idx] += flow
        
        return edge_flows
