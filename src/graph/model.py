"""Модуль с классами для представления графа электрической сети."""

class Node:
    """Вершина графа."""
    def __init__(self, name):
        self.name = name
        self.type = self._determine_type(name)
        self.edges = []

    @staticmethod
    def _determine_type(name):
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
    def __init__(self, node1, node2, capacity):
        self.nodes = (node1, node2)
        self.capacity = capacity
        node1.edges.append(self)
        node2.edges.append(self)


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, name):
        if name not in self.nodes:
            self.nodes[name] = Node(name)
        return self.nodes[name]

    def add_edge(self, name1, name2, capacity):
        n1 = self.add_node(name1)
        n2 = self.add_node(name2)
        edge = Edge(n1, n2, capacity)
        self.edges.append(edge)
        return edge

    def get_node(self, name):
        """Возвращает вершину по имени или None, если не найдена."""
        return self.nodes.get(name)

    def get_sources(self):
        """Возвращает список всех вершин-источников (source)."""
        return [node for node in self.nodes.values() if node.type == 'source']

    def get_consumers(self):
        """Возвращает список всех вершин-потребителей (consumer)."""
        return [node for node in self.nodes.values() if node.type == 'consumer']