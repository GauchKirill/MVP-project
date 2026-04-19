"""Основной модуль для работы с графом электрической сети «Альфа»."""

import json
from graph import Graph, GraphView
from flow import FlowTask

SETTING_FOLDER: str = 'settings'
EDGES_FILE: str = 'edges.json'
GENERETED_FOLDER: str = 'genereted'
GRAPH_FILE: str = 'graph.html'


def load_edges(graph, filename):
    """Загружает рёбра графа из JSON-файла."""
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


def main():
    """Главная функция."""
    print("=" * 60)
    print("ЗАГРУЗКА ГРАФА ЭЛЕКТРИЧЕСКОЙ СЕТИ «АЛЬФА»")
    print("=" * 60)
    
    # 1. Загружаем граф
    graph = Graph()
    load_edges(graph, f"{SETTING_FOLDER}/{EDGES_FILE}")
    print(f"✓ Граф загружен: {len(graph.nodes)} вершин, {len(graph.edges)} рёбер")
    
    # 2. Создаём задачу и генерируем все пары источник-потребитель
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ ПАР ИСТОЧНИК-ПОТРЕБИТЕЛЬ")
    print("=" * 60)
    
    task = FlowTask(graph)
    pairs_count = task.generate_all_pairs()
    print(f"✓ Сгенерировано пар: {pairs_count}")
    
    # 3. Строим все возможные пути
    print("\n" + "=" * 60)
    print("ПОСТРОЕНИЕ ПУТЕЙ")
    print("=" * 60)
    
    task.build_all_paths(max_depth=50)
    
    # 4. Статистика
    print("\n" + "=" * 60)
    print("СТАТИСТИКА")
    print("=" * 60)
    
    stats = task.get_statistics()
    print(f"Источников: {stats['total_sources']}")
    print(f"Потребителей: {stats['total_consumers']}")
    print(f"Пар всего: {stats['total_flows']}")
    print(f"Пар с путями: {stats['flows_with_paths']}")
    print(f"Пар без путей: {stats['flows_without_paths']}")
    
    # 5. Выводим сводку по всем путям
    task.print_all_paths_summary(max_per_flow=3)
    
    # 6. Пример детального вывода для одного потока (для отладки)
    if task.flows:
        # Выбираем поток с путями для примера
        sample_flow = next((f for f in task.flows if f.paths), None)
        if sample_flow:
            print("\n" + "=" * 60)
            print("ПРИМЕР ДЕТАЛЬНОГО ВЫВОДА ПУТЕЙ")
            print("=" * 60)
            task.print_flow_paths(sample_flow, max_display=10)
    
    # 7. Визуализация графа (опционально)
    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ")
    print("=" * 60)
    
    view = GraphView(graph)
    view.draw_pyvis(filename=f"{GENERETED_FOLDER}/{GRAPH_FILE}")
    
    print("\n✓ Готово!")


if __name__ == "__main__":
    main()