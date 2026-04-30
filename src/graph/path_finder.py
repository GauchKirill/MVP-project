from typing import List, Set
from .model import Graph, Node, Edge


class PathFinder:
    """Класс для поиска путей в графе"""
    def __init__(self, graph: Graph):
        self.graph = graph
    
    def find_all_simple_paths(self, source: Node, target: Node) -> List[List[Edge]]:
        """
        находит все простые пути (без циклов) от source до target
        """
        paths = []
        visited_nodes: Set[str] = set()
        current_path: List[Edge] = []
        
        self._dfs_paths(source, target, visited_nodes, current_path, paths)
        
        return paths
    
    def _dfs_paths(self, 
                   current_node: Node, 
                   target: Node, 
                   visited_nodes: Set[str], 
                   current_path: List[Edge], 
                   all_paths: List[List[Edge]]):
        """
        реализовал рекурсивный поиск путей с помощью DFS
        """
        
        if current_node == target:
            if current_path:
                all_paths.append(current_path.copy())
            return
        
        visited_nodes.add(current_node.name)
        
        for edge in current_node.edges:
            next_node = edge.nodes[0] if edge.nodes[1] == current_node else edge.nodes[1]
            
            if next_node.name not in visited_nodes:
                current_path.append(edge)
                self._dfs_paths(next_node, target, visited_nodes, current_path, 
                               all_paths)
                current_path.pop()
        
        visited_nodes.remove(current_node.name)
    
    def find_paths_with_capacity_constraint(self, 
                                           source: Node, 
                                           target: Node, 
                                           min_capacity: float) -> List[List[Edge]]:
        """
        находит пути, где каждое ребро имеет пропускную способность >= min_capacity
        """
        all_paths = self.find_all_simple_paths(source, target)
        
        valid_paths = []
        for path in all_paths:
            if all(edge.capacity >= min_capacity for edge in path):
                valid_paths.append(path)
        
        return valid_paths
    
    def get_path_capacity(self, path: List[Edge]) -> float:
        """
        возвращает пропускную способность пути (минимум по всем рёбрам)
        """
        if not path:
            return 0.0
        return min(edge.capacity for edge in path)
    
    def get_path_length(self, path: List[Edge]) -> int:
        """
        возвращает длину пути в количестве рёбер
        """
        return len(path)