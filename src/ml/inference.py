import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

from .model import PathWeightNetwork
from .feature_extractor import FeatureExtractor
from .loss import EdgeFlowCalculator


class FlowPredictor:
    """
    Класс для предсказания распределения потоков с помощью обученной модели
    """
    def __init__(self,
                 model: PathWeightNetwork,
                 feature_extractor: FeatureExtractor,
                 edge_calculator: EdgeFlowCalculator,
                 device: str = 'cpu'):
        
        self.model = model.to(device)
        self.feature_extractor = feature_extractor
        self.edge_calculator = edge_calculator
        self.device = device
        
        self.model.eval()
    
    def predict_with_normalized(self, 
                                normalized_features: np.ndarray,
                                flows: Dict[str, Dict[str, float]],
                                capacity_mask: np.ndarray) -> Dict:
        """
        Предсказывает распределение потоков, используя уже нормализованные признаки
        
        Args:
            normalized_features: (feature_dim,) — нормализованный вектор признаков
            flows: словарь заявок {source: {consumer: demand}} — в кВт
            capacity_mask: (E,) — 1.0 для конечных capacity, 0.0 для inf
            
        Returns:
            словарь с результатами:
                - path_weights: (S, C, max_paths) — веса распределения
                - path_flows: (S, C, max_paths) — потоки по путям
                - edge_flows: (E,) — суммарные потоки на рёбрах
                - edge_utilization: (E,) — загрузка рёбер (0..1)
                - total_delivered: суммарная доставка
                - demanded: суммарная заявка
        """
        features_tensor = torch.from_numpy(normalized_features).unsqueeze(0).float().to(self.device)
        demands = self._create_demand_matrix(flows)
        demands_tensor = torch.from_numpy(demands).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            path_weights = self.model(features_tensor)[0]  # (S, C, max_paths)
            # Последний путь — фиктивный, исключаем его из реальных весов
            real_weights = path_weights[:, :, :self.feature_extractor.max_paths]  # (S, C, max_real_paths)
            path_flows = real_weights * demands_tensor[0].unsqueeze(-1)
            edge_flows = self.edge_calculator.compute_edge_flows(path_flows.unsqueeze(0))[0]
        
        # В numpy
        path_weights_np = real_weights.cpu().numpy()
        path_flows_np = path_flows.cpu().numpy()
        edge_flows_np = edge_flows.cpu().numpy()
        # Веса фиктивного пути (доля потерь)
        loss_weights_np = path_weights[:, :, -1].cpu().numpy()  # (S, C)
        
        # Загрузка рёбер
        edge_capacities = self.feature_extractor.get_edge_capacities()
        edge_utilization = np.zeros_like(edge_flows_np)
        finite_mask = capacity_mask > 0.5
        if finite_mask.any():
            edge_utilization[finite_mask] = np.divide(
                edge_flows_np[finite_mask],
                edge_capacities[finite_mask],
                out=np.zeros_like(edge_flows_np[finite_mask]),
                where=edge_capacities[finite_mask] < 1e8
            )
        
        # Суммарная доставка
        delivered_np = path_flows_np.sum(axis=-1)
        total_delivered = delivered_np.sum()
        total_demanded = demands.sum()
        
        return {
            'path_weights': path_weights_np,
            'path_flows': path_flows_np,
            'edge_flows': edge_flows_np,
            'edge_utilization': edge_utilization,
            'loss_weights': loss_weights_np,
            'total_delivered': total_delivered,
            'demanded': total_demanded
        }
    
    def _create_demand_matrix(self, flows: Dict) -> np.ndarray:
        """
        Создаёт матрицу заявок (S, C) из словаря {source: {consumer: demand}}
        """
        S = self.feature_extractor.S
        C = self.feature_extractor.C
        demands = np.zeros((S, C), dtype=np.float32)
        
        for s_name, consumers in flows.items():
            if s_name not in self.feature_extractor.source_to_idx:
                continue
            s_idx = self.feature_extractor.source_to_idx[s_name]
            for c_name, demand in consumers.items():
                if c_name not in self.feature_extractor.consumer_to_idx:
                    continue
                c_idx = self.feature_extractor.consumer_to_idx[c_name]
                demands[s_idx, c_idx] = demand
        
        return demands
    
    def predict_batch(self, flows_list: List[Dict]) -> List[Dict]:
        """
        Предсказывает для батча сценариев (каждый сценарий нормализуется индивидуально)
        """
        results = []
        for flows in flows_list:
            raw = self.feature_extractor.build_raw_features(flows)
            norm_features, cap_mask = self.feature_extractor.normalize_features(raw)
            results.append(self.predict_with_normalized(norm_features, flows, cap_mask))
        return results
    
    def save_results(self, results: Dict, filename: str):
        """Сохраняет результаты в JSON"""
        serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            else:
                serializable[key] = value
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)