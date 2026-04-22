"""Функция потерь для обучения модели распределения потоков."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class PowerFlowLoss(nn.Module):
    """
    Функция потерь для оптимизации потокораспределения.
    Работает с нормализованными значениями в диапазоне [0, 1].
    """
    
    def __init__(self,
                 capacity_weight: float = 1.0,
                 demand_weight: float = 1.0):
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
                path_lengths: Optional[torch.Tensor] = None,
                path_masks: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Вычисляет функцию потерь с нормализацией относительно целевых значений.
        
        loss_capacity = Σ max(0, (flow - capacity) / capacity)²
        loss_demand = Σ ((delivered - demand) / demand)²  для недопоставки
        """
        components = {}
        total_loss = torch.tensor(0.0, device=path_flows.device)
        
        # 1. Capacity loss — нормализованный штраф за превышение пропускной способности
        edge_capacities_expanded = edge_capacities.unsqueeze(0)  # (1, E)
        
        # Превышение в абсолютных величинах
        capacity_violation = F.relu(edge_flows - edge_capacities_expanded)  # (batch, E)
        
        # Нормализация: делим на capacity (избегаем деления на 0)
        normalized_violation = capacity_violation / edge_capacities_expanded.clamp(min=1e-8)
        
        # Квадрат относительного превышения, сумма по рёбрам для каждого сценария
        violation_per_scenario = (normalized_violation ** 2).sum(dim=1)  # (batch,)
        
        # Среднее по батчу
        loss_capacity = self.capacity_weight * violation_per_scenario.mean()
        
        total_loss = total_loss + loss_capacity
        components['capacity'] = loss_capacity.item()
        
        # 2. Demand loss — нормализованный штраф за недопоставку
        delivered = path_flows.sum(dim=-1)  # (batch, S, C)

        # Недопоставка в абсолютных величинах
        shortage = demands - delivered  # (batch, S, C)
        
        # Нормализация: делим на demand (избегаем деления на 0)
        normalized_shortage = shortage / demands.clamp(min=1e-8)
        
        # Квадрат относительной недопоставки, сумма по всем парам (S*C) для каждого сценария
        shortage_per_scenario = (normalized_shortage ** 2).sum(dim=(1, 2))  # (batch,)
        
        # Среднее по батчу
        loss_demand = self.demand_weight * shortage_per_scenario.mean()
        
        total_loss = total_loss + loss_demand
        components['demand'] = loss_demand.item()
        
        # Логируем дополнительные метрики
        with torch.no_grad():
            # # Средняя загрузка рёбер (в долях от capacity)
            # edge_utils = edge_flows / edge_capacities_expanded.clamp(min=1e-8)
            # components['avg_utilization'] = edge_utils.mean().item()
             
            # # Доля доставленной энергии
            # total_demanded = demands.sum()
            # total_delivered = delivered.sum()
            # if total_demanded > 0:
            #     components['delivery_ratio'] = (total_delivered / total_demanded).item()
            # else:
            #     components['delivery_ratio'] = 1.0
            # Маска конечных рёбер (capacity < 1e8 считаем конечной)
            finite_mask = edge_capacities < 1e8
            
            # Абсолютная загрузка рёбер (для информации)
            if finite_mask.any():
                edge_utils_finite = edge_flows[:, finite_mask] / edge_capacities[finite_mask].clamp(min=1e-8)
                components['avg_utilization'] = edge_utils_finite.mean().item()
                components['max_utilization'] = edge_utils_finite.max().item()
                
                # Максимальное относительное превышение (для отладки)
                if (capacity_violation[:, finite_mask] > 0).any():
                    max_relative_violation = normalized_violation[:, finite_mask].max().item()
                    components['max_relative_violation'] = max_relative_violation
            else:
                components['avg_utilization'] = 0.0
                components['max_utilization'] = 0.0
            
            # Доля доставленной энергии
            total_demanded = demands.sum(dim=(1, 2))  # (batch,)
            total_delivered = delivered.sum(dim=(1, 2))  # (batch,)
            delivery_ratios = total_delivered / total_demanded.clamp(min=1e-8)
            components['delivery_ratio'] = delivery_ratios.mean().item()
            
            # Средние значения нормализованных отклонений (для отладки)
            components['avg_normalized_violation'] = normalized_violation.mean().item()
            components['avg_normalized_shortage'] = normalized_shortage.mean().item()
            
            # Количество сценариев с нарушениями
            components['scenarios_with_violation'] = (violation_per_scenario > 0).sum().item()
            components['scenarios_with_shortage'] = (shortage_per_scenario > 0).sum().item()
        
        self.last_losses = components
        return total_loss, components
    
    def get_last_losses(self) -> Dict[str, float]:
        """Возвращает компоненты функции потерь с последнего вызова."""
        return self.last_losses.copy()


class EdgeFlowCalculator:
    """
    Вычисляет суммарные потоки на рёбрах на основе потоков по путям.
    """
    
    def __init__(self, registry, feature_extractor):
        """
        Args:
            registry: RequestRegistry с информацией о путях
            feature_extractor: FeatureExtractor с информацией о рёбрах
        """
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
    
    def compute_edge_flows(self, path_flows: torch.Tensor) -> torch.Tensor:
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
    
    def compute_edge_flows_normalized(self, 
                                      path_flows: torch.Tensor,
                                      normalizer: float = 1.0) -> torch.Tensor:
        """
        Вычисляет нормализованные потоки на рёбрах.
        
        Args:
            path_flows: тензор (batch_size, S, C, max_paths) - нормализованные потоки
            normalizer: нормировочный коэффициент
            
        Returns:
            тензор (batch_size, E) с нормализованными суммарными потоками
        """
        # Просто вызываем обычный метод, т.к. path_flows уже нормализованы
        return self.compute_edge_flows(path_flows)