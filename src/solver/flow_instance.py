"""Класс для хранения текущего распределения потока по путям заявки."""

from typing import Dict, List, Optional
from graph import Request, Edge


class FlowInstance:
    """Представляет текущее распределение потока для конкретной заявки."""
    
    def __init__(self, request: Request, target_amount: float):
        self.request = request
        self.target_amount = target_amount  # требуемая мощность из flows.json
        
        # Словарь: путь (кортеж id рёбер) -> текущая мощность
        # Для однозначной идентификации пути используем tuple(id(edge))
        self.path_flows: Dict[tuple, float] = {}
        
        # Инициализируем все пути нулевым потоком
        for path in request.paths:
            path_key = self._path_to_key(path)
            self.path_flows[path_key] = 0.0
    
    @staticmethod
    def _path_to_key(path: List[Edge]) -> tuple:
        """Преобразует путь в кортеж id рёбер для использования в словаре."""
        return tuple(id(edge) for edge in path)
    
    def set_uniform_flow(self):
        """Равномерно распределяет целевой поток по всем доступным путям."""
        if not self.request.paths:
            return
        per_path = self.target_amount / len(self.request.paths)
        for path in self.request.paths:
            key = self._path_to_key(path)
            self.path_flows[key] = per_path
    
    def get_total_flow(self) -> float:
        """Возвращает суммарный поток по всем путям данной заявки."""
        return sum(self.path_flows.values())
    
    def get_path_flow(self, path: List[Edge]) -> float:
        """Возвращает поток по конкретному пути."""
        key = self._path_to_key(path)
        return self.path_flows.get(key, 0.0)
    
    def update_path_flow(self, path: List[Edge], delta: float):
        """Изменяет поток по пути на delta (с ограничением неотрицательности)."""
        key = self._path_to_key(path)
        if key in self.path_flows:
            new_val = self.path_flows[key] + delta
            if new_val < 0:
                new_val = 0.0
            self.path_flows[key] = new_val
    
    def get_paths(self) -> List[List[Edge]]:
        """Возвращает список путей данной заявки."""
        return self.request.paths