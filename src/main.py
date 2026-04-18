"""Основной модуль для работы с графом электрической сети «Альфа»."""

import json
from graph import Graph, GraphView

SETTING_FOLDER: str = 'settings'
EDGES_FILE: str = 'edges.json'
GENERETED_FOLDER: str = 'genereted'
GRAPH_FILE: str = 'graph.html'


def load_edges(graph, filename):
    with open(filename, 'r') as f:
        edges_data = json.load(f)
    for item in edges_data:
        n1, n2 = item['nodes']
        cap = item['capacity']
        if cap == 'inf':
            cap = float('inf')
        else:
            cap = float(cap)
        graph.add_edge(n1, n2, cap)


if __name__ == "__main__":
    graph = Graph()
    load_edges(graph, SETTING_FOLDER + "/" + EDGES_FILE)
    print(f"Граф загружен: {len(graph.nodes)} вершин, {len(graph.edges)} рёбер")
    
    view = GraphView(graph)
    view.draw_pyvis(filename=GENERETED_FOLDER + "/" + GRAPH_FILE)