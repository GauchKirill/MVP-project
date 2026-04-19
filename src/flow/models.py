"""Модели для представления потоков."""

from typing import List, Optional
from graph.model import Node, Edge


class Flow:
    """Класс представляющий требуемый поток от источника к потребителю."""
    
    def __init__(self, source: Node, consumer: Node, amount: float):
        self.source = source
        self.consumer = consumer
        self.amount = amount
        self.paths: List[List[Edge]] = []  # список всех возможных путей
        
    def add_path(self, path: List[Edge]):
        """Добавляет путь к списку возможных путей."""
        self.paths.append(path)
        
    def get_paths_count(self) -> int:
        """Возвращает количество найденных путей."""
        return len(self.paths)
    
    def get_shortest_path(self) -> Optional[List[Edge]]:
        """Возвращает самый короткий путь (по количеству рёбер)."""
        if not self.paths:
            return None
        return min(self.paths, key=len)
    
    def get_longest_path(self) -> Optional[List[Edge]]:
        """Возвращает самый длинный путь (по количеству рёбер)."""
        if not self.paths:
            return None
        return max(self.paths, key=len)
    
    def __str__(self) -> str:
        return f"Flow({self.source.name} -> {self.consumer.name}, amount={self.amount})"
    
    def __repr__(self) -> str:
        return self.__str__()