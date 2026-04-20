import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

from .model import PathWeightNetwork
from .feature_extractor import FeatureExtractor
from .loss import EdgeFlowCalculator


class FlowPredictor:
    """
    Класс для предсказания распределения потоков с помощью обученной модели.
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
    
    def predict(self, flows: Dict[str, Dict[str, float]]) -> Dict:
        """
        Предсказывает распределение потоков для заданных заявок.
        
        Args:
            flows: словарь заявок вида {source: {consumer: demand}}
            
        Returns:
            словарь с результатами:
                - path_weights: веса путей (S, C, max_paths)
                - path_flows: потоки по путям (S, C, max_paths)
                - edge_flows: суммарные потоки на рёбрах (E,)
                - edge_utilization: загрузка рёбер в %
                - total_delivered: суммарно доставленная мощность
        """
        # Извлекаем признаки
        features = self.feature_extractor.extract_features(flows)
        features_tensor = torch.FloatTensor([features]).to(self.device)
        
        # Создаём матрицу заявок
        demands = self._create_demand_matrix(flows)
        demands_tensor = torch.FloatTensor([demands]).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            path_weights = self.model(features_tensor)[0]  # (S, C, max_paths)
            path_flows = path_weights * demands_tensor[0].unsqueeze(-1)
            edge_flows = self.edge_calculator.compute_edge_flows(path_flows.unsqueeze(0))[0]
        
        # Конвертируем в numpy
        path_weights_np = path_weights.cpu().numpy()
        path_flows_np = path_flows.cpu().numpy()
        edge_flows_np = edge_flows.cpu().numpy()
        
        # Вычисляем загрузку рёбер
        edge_capacities = self.feature_extractor.get_edge_capacities()
        edge_utilization = np.divide(
            edge_flows_np,
            edge_capacities,
            out=np.zeros_like(edge_flows_np),
            where=edge_capacities < 1e8
        )
        
        # Суммарная доставка
        demands_np = demands.flatten()
        delivered_np = path_flows_np.sum(axis=-1).flatten()
        total_delivered = min(demands_np.sum(), delivered_np.sum())
        
        return {
            'path_weights': path_weights_np,
            'path_flows': path_flows_np,
            'edge_flows': edge_flows_np,
            'edge_utilization': edge_utilization,
            'total_delivered': total_delivered,
            'demanded': demands_np.sum()
        }
    
    def _create_demand_matrix(self, flows: Dict) -> np.ndarray:
        """Создаёт матрицу заявок (S, C)."""
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
        """Предсказывает для батча сценариев."""
        results = []
        for flows in flows_list:
            results.append(self.predict(flows))
        return results
    
    def save_results(self, results: Dict, filename: str):
        """Сохраняет результаты в JSON."""
        # Конвертируем numpy массивы в списки для JSON
        serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            else:
                serializable[key] = value
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
