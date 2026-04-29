from typing import List, Dict, Optional, Tuple

class Node:
    """Класс вершины графа"""
    def __init__(self, name: str):
        self.name: str = name
        self.type: str = self._determine_type(name)
        self.edges: List['Edge'] = []

    @staticmethod
    def _determine_type(name: str) -> str:
        if name.isalpha() and name.isupper() and len(name) == 1 and name.isascii():
            return "source"
        if name.isdigit():
            return "consumer"
        if name.endswith('.') and all(c in "IVXLCDM" for c in name[:-1]):
            return "junction"
        if name.startswith('v') and name[1:].isdigit():
            return "additional"
        return "unknown"


class Edge:
    """Класс ребра графа"""
    def __init__(self, node1: Node, node2: Node, capacity: float):
        self.nodes: Tuple[Node, Node] = (node1, node2)
        self.capacity: float = capacity
        node1.edges.append(self)
        node2.edges.append(self)


class Graph:
    """Граф электрической сети"""
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []

    def add_node(self, name: str) -> Node:
        """добавляет вершину в граф"""
        if name not in self.nodes:
            self.nodes[name] = Node(name)
        return self.nodes[name]

    def add_edge(self, name1: str, name2: str, capacity: float) -> Edge:
        """добавляет ребро между вершинами"""
        n1 = self.add_node(name1)
        n2 = self.add_node(name2)
        edge = Edge(n1, n2, capacity)
        self.edges.append(edge)
        return edge

    def get_node(self, name: str) -> Optional[Node]:
        """возвращает вершину по имени (None, если не найдена)"""
        return self.nodes.get(name)
    
    def has_node(self, name: str) -> bool:
        """проверяет существование вершины в графе"""
        return name in self.nodes

    def get_sources(self) -> List[Node]:
        """возвращает список всех вершин-источников"""
        return [node for node in self.nodes.values() if node.type == 'source']

    def get_consumers(self) -> List[Node]:
        """возвращает список всех вершин-потребителей"""
        return [node for node in self.nodes.values() if node.type == 'consumer']