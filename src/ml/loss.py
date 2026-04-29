import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class PowerFlowLoss(nn.Module):
    """
    Функция потерь для оптимизации потокораспределения
    
    - loss_capacity: штраф за превышение пропускной способности (только для конечных capacity)
    - loss_demand: штраф за -недо и -пере поставку относительно заявок
    """
    def __init__(self,
                 capacity_weight: float = 1.0,
                 demand_weight: float = 1.0):
        super().__init__()
        self.capacity_weight = capacity_weight
        self.demand_weight = demand_weight
        self.last_losses: Dict[str, float] = {}
    
    def forward(self,
                path_flows: torch.Tensor,
                edge_flows: torch.Tensor,
                demands: torch.Tensor,
                edge_capacities: torch.Tensor,
                capacity_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            - edge_capacities: (batch_size, E) — нормализованные capacity из признаков,
                            где inf заменены на 1.0
            - capacity_mask: (batch_size, E) — 1.0 для конечных capacity, 0.0 для бывших inf
        """
        components = {}
        total_loss = torch.tensor(0.0, device=path_flows.device)
        
        # Применяем маску: для inf-рёбер зануляем и поток, и capacity
        if capacity_mask is not None:
            edge_flows_masked = edge_flows * capacity_mask
            edge_capacities_masked = edge_capacities * capacity_mask
        else:
            edge_flows_masked = edge_flows
            edge_capacities_masked = edge_capacities
        
        # capacity loss — превышение над замаскированной capacity
        capacity_violation = F.relu(edge_flows_masked - edge_capacities_masked)
        

        loss_capacity = self.capacity_weight * capacity_violation.mean()
        
        total_loss = total_loss + loss_capacity
        components['capacity'] = loss_capacity.item()
        
        # demand loss
        delivered = path_flows.sum(dim=-1)
        shortage = torch.abs(demands - delivered)
        loss_demand = self.demand_weight * shortage.mean()
        total_loss = total_loss + loss_demand
        components['demand'] = loss_demand.item()
        
        # метрики
        with torch.no_grad():
            if capacity_mask is not None and capacity_mask.sum() > 0:
                edge_utils = edge_flows_masked / (edge_capacities_masked + 1e-8)
                edge_utils = edge_utils[capacity_mask > 0.5]
                if edge_utils.numel() > 0:
                    components['avg_util'] = edge_utils.mean().item()
                    components['max_util'] = edge_utils.max().item()
                    components['overloaded'] = (edge_utils > 1.0).sum().item()
                else:
                    components['avg_util'] = 0.0
                    components['max_util'] = 0.0
                    components['overloaded'] = 0
            else:
                components['avg_util'] = 0.0
                components['max_util'] = 0.0
                components['overloaded'] = 0
        
        self.last_losses = components
        return total_loss, components


class EdgeFlowCalculator:
    """
    Вычисляет суммарные потоки на рёбрах на основе потоков по путям
    """
    
    def __init__(self, registry, feature_extractor):
        self.registry = registry
        self.extractor = feature_extractor
        self.path_to_edges = self._build_path_to_edges_mapping()
    
    def _build_path_to_edges_mapping(self) -> Dict[Tuple[int, int, int], List[int]]:
        """Строит отображение (s_idx, c_idx, path_idx) → список индексов рёбер"""
        mapping = {}
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
        Вычисляет суммарные потоки на рёбрах
        
        Args:
            path_flows: (batch_size, S, C, max_paths)
            
        Returns:
            (batch_size, E)
        """
        batch_size = path_flows.shape[0]
        E = self.extractor.E
        device = path_flows.device
        edge_flows = torch.zeros(batch_size, E, device=device)
        
        for (s_idx, c_idx, path_idx), edge_indices in self.path_to_edges.items():
            flow = path_flows[:, s_idx, c_idx, path_idx]
            for edge_idx in edge_indices:
                edge_flows[:, edge_idx] += flow
        
        return edge_flows