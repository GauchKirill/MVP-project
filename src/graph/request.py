from typing import List, Optional
from .model import Node, Edge


class Request:
    """Заявка на передачу электроэнергии от источника к потребителю"""
    def __init__(self, source: Node, consumer: Node):
        self.source = source
        self.consumer = consumer
        self.paths: List[List[Edge]] = []  # список всех возможных путей
        
    def add_path(self, path: List[Edge]):
        """добавляет путь к списку возможных путей."""
        self.paths.append(path)
        
    def get_paths_count(self) -> int:
        """возвращает количество найденных путей"""
        return len(self.paths)
    
    def get_shortest_path(self) -> Optional[List[Edge]]:
        """возвращает самый короткий путь (по количеству рёбер)"""
        if not self.paths:
            return None
        return min(self.paths, key=len)
    
    def get_longest_path(self) -> Optional[List[Edge]]:
        """возвращает самый длинный путь (по количеству рёбер)"""
        if not self.paths:
            return None
        return max(self.paths, key=len)
    
    def __str__(self) -> str:
        return f"Request({self.source.name} -> {self.consumer.name})"
    
    def __repr__(self) -> str:
        return self.__str__()