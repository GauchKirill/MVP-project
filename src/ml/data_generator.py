# src/ml/data_generator.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
from scipy.stats import qmc


class DataGenerator:
    """
    Генератор обучающих данных с использованием Latin Hypercube Sampling
    и вариаций реальных данных.
    """
    
    def __init__(self, 
        base_flows: Dict[str, Dict[str, float]],
        feature_dim: int,
        sources: List[str],
        consumers: List[str]):
        self.base_flows = base_flows
        self.feature_dim = feature_dim
        self.sources = sources
        self.consumers = consumers
        
        # Собираем список всех возможных пар (source, consumer)
        self.all_pairs = []
        for s in sources:
            for c in consumers:
                self.all_pairs.append((s, c))
    
    def generate_lhs_samples(self, 
                            num_samples: int, 
                            sparsity: float = 0.7) -> List[Dict]:
        """
        Генерирует сценарии методом Latin Hypercube Sampling.
        
        Args:
            num_samples: количество сценариев
            sparsity: доля нулевых заявок (разреженность)
            
        Returns:
            список словарей с заявками
        """
        # Используем LHS для равномерного покрытия гиперкуба
        sampler = qmc.LatinHypercube(d=self.feature_dim)
        samples = sampler.random(n=num_samples)
        
        scenarios = []
        for sample in samples:
            flows = self._sample_to_flows(sample, sparsity)
            scenarios.append(flows)
        
        return scenarios
    
    def generate_varied_samples(self, 
                                num_samples: int, 
                                noise_std: float = 0.3) -> List[Dict]:
        """
        Генерирует сценарии путём добавления шума к реальным данным.
        
        Args:
            num_samples: количество сценариев
            noise_std: стандартное отклонение шума (в долях от значения)
            
        Returns:
            список словарей с заявками
        """
        scenarios = []
        
        for _ in range(num_samples):
            flows = deepcopy(self.base_flows)
            
            for source in flows:
                for consumer in flows[source]:
                    base_value = flows[source][consumer]
                    # Логнормальный шум (чтобы не уходить в отрицательные значения)
                    noise = np.random.lognormal(mean=0, sigma=noise_std)
                    new_value = base_value * noise
                    
                    # Ограничиваем снизу
                    flows[source][consumer] = max(0.01 * base_value, new_value)
            
            scenarios.append(flows)
        
        return scenarios
    
    def generate_mixed_samples(self, 
                            total_samples: int,
                            lhs_ratio: float = 0.3,
                            sparsity: float = 0.7,
                            noise_std: float = 0.3) -> List[Dict]:
        """
        Генерирует смешанную выборку: LHS + вариации реальных данных.
        
        Args:
            total_samples: общее количество сценариев
            lhs_ratio: доля LHS сэмплов
            sparsity: разреженность для LHS
            noise_std: уровень шума для вариаций
            
        Returns:
            список словарей с заявками
        """
        num_lhs = int(total_samples * lhs_ratio)
        num_varied = total_samples - num_lhs
        
        scenarios = []
        
        if num_lhs > 0:
            lhs_scenarios = self.generate_lhs_samples(num_lhs, sparsity)
            scenarios.extend(lhs_scenarios)
        
        if num_varied > 0:
            varied_scenarios = self.generate_varied_samples(num_varied, noise_std)
            scenarios.extend(varied_scenarios)
        
        # Перемешиваем
        np.random.shuffle(scenarios)
        
        return scenarios
    
    def _sample_to_flows(self, sample: np.ndarray, sparsity: float) -> Dict:
        """
        Преобразует точку из гиперкуба в словарь заявок.
        
        Args:
            sample: вектор размера feature_dim в диапазоне [0, 1]
            sparsity: доля нулевых заявок
            
        Returns:
            словарь с заявками
        """
        flows = {s: {} for s in self.sources}
        
        # Определяем, сколько признаков соответствует заявкам
        # (предполагаем, что первые E признаков — capacity рёбер,
        #  остальные — заявки S*C)
        demand_features = sample[-len(self.all_pairs):]
        
        # Применяем разреженность
        mask = np.random.random(len(demand_features)) > sparsity
        demand_features = demand_features * mask
        
        # Масштабируем заявки (умножаем на типичную величину)
        typical_demand = self._get_typical_demand()
        demand_features = demand_features * typical_demand * 2
        
        # Распределяем по парам
        for idx, (s_name, c_name) in enumerate(self.all_pairs):
            if idx < len(demand_features) and demand_features[idx] > 0:
                flows[s_name][c_name] = float(demand_features[idx])
        
        return flows
    
    def _get_typical_demand(self) -> float:
        """Возвращает типичную величину заявки (медиану по всем ненулевым)."""
        all_demands = []
        for consumers in self.base_flows.values():
            all_demands.extend(consumers.values())
        
        if all_demands:
            return float(np.median(all_demands))
        return 100.0  # fallback


class PhysicsValidator:
    """
    Проверяет физическую реализуемость сгенерированных сценариев.
    """
    
    def __init__(self, graph, feature_extractor):
        self.graph = graph
        self.extractor = feature_extractor
    
    def is_feasible(self, flows: Dict) -> bool:
        """
        Проверяет, не нарушает ли сценарий базовые физические ограничения.
        
        Критерии:
        1. Сумма заявок от одного источника не превышает разумного предела
        2. Нет отрицательных заявок
        """
        # Проверка на отрицательные значения
        for source, consumers in flows.items():
            for demand in consumers.values():
                if demand < 0:
                    return False
        
        # Можно добавить дополнительные проверки
        return True
    
    def filter_feasible(self, scenarios: List[Dict]) -> List[Dict]:
        """Оставляет только физически реализуемые сценарии."""
        return [s for s in scenarios if self.is_feasible(s)]
