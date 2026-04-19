"""Модуль для поиска всех простых путей между вершинами графа."""

from typing import List, Set, Optional
from graph.model import Graph, Node, Edge


class PathFinder:
    """Класс для поиска путей в графе."""
    
    def __init__(self, graph: Graph):
        self.graph = graph
    
    def find_all_simple_paths(self, source: Node, target: Node, max_depth: int = 50) -> List[List[Edge]]:
        """
        Находит все простые пути (без циклов) от source до target.
        
        Args:
            source: начальная вершина
            target: конечная вершина
            max_depth: максимальная глубина поиска (защита от бесконечной рекурсии)
            
        Returns:
            Список всех найденных путей, где каждый путь - список рёбер
        """
        paths = []
        visited_nodes: Set[str] = set()
        current_path: List[Edge] = []
        
        self._dfs_paths(source, target, visited_nodes, current_path, paths, max_depth, depth=0)
        
        return paths
    
    def _dfs_paths(self, 
                   current_node: Node, 
                   target: Node, 
                   visited_nodes: Set[str], 
                   current_path: List[Edge], 
                   all_paths: List[List[Edge]], 
                   max_depth: int,
                   depth: int):
        """
        Рекурсивный поиск путей с помощью DFS.
        
        Args:
            current_node: текущая вершина
            target: целевая вершина
            visited_nodes: множество посещённых вершин
            current_path: текущий путь (список рёбер)
            all_paths: список всех найденных путей
            max_depth: максимальная глубина
            depth: текущая глубина
        """
        # Защита от бесконечной рекурсии
        if depth > max_depth:
            return
        
        # Если достигли цели
        if current_node == target:
            if current_path:  # путь не пустой
                all_paths.append(current_path.copy())
            return
        
        # Помечаем текущую вершину как посещённую
        visited_nodes.add(current_node.name)
        
        # Исследуем все инцидентные рёбра
        for edge in current_node.edges:
            # Определяем соседнюю вершину
            next_node = edge.nodes[0] if edge.nodes[1] == current_node else edge.nodes[1]
            
            # Если вершина ещё не посещена
            if next_node.name not in visited_nodes:
                current_path.append(edge)
                self._dfs_paths(next_node, target, visited_nodes, current_path, 
                               all_paths, max_depth, depth + 1)
                current_path.pop()
        
        # Снимаем пометку о посещении (backtracking)
        visited_nodes.remove(current_node.name)
    
    def find_paths_with_capacity_constraint(self, 
                                           source: Node, 
                                           target: Node, 
                                           min_capacity: float) -> List[List[Edge]]:
        """
        Находит пути, где каждое ребро имеет пропускную способность >= min_capacity.
        
        Args:
            source: начальная вершина
            target: конечная вершина
            min_capacity: минимальная допустимая пропускная способность
            
        Returns:
            Список путей, удовлетворяющих ограничению
        """
        all_paths = self.find_all_simple_paths(source, target)
        
        valid_paths = []
        for path in all_paths:
            if all(edge.capacity >= min_capacity for edge in path):
                valid_paths.append(path)
        
        return valid_paths
    
    def get_path_capacity(self, path: List[Edge]) -> float:
        """
        Возвращает пропускную способность пути (минимум по всем рёбрам).
        
        Args:
            path: список рёбер пути
            
        Returns:
            Минимальная пропускная способность среди всех рёбер пути
        """
        if not path:
            return 0.0
        return min(edge.capacity for edge in path)
    
    def get_path_length(self, path: List[Edge]) -> int:
        """
        Возвращает длину пути в количестве рёбер.
        
        Args:
            path: список рёбер пути
            
        Returns:
            Количество рёбер в пути
        """
        return len(path)