"""Основной модуль для работы с графом электрической сети «Альфа»."""

import json
from graph import Graph, GraphView, RequestRegistry

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
    
    # 2. Создаём реестр заявок и генерируем все пары источник-потребитель
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ ЗАЯВОК (ВСЕ ПАРЫ ИСТОЧНИК-ПОТРЕБИТЕЛЬ)")
    print("=" * 60)
    
    registry = RequestRegistry(graph)
    requests_count = registry.generate_all_requests()
    print(f"✓ Сгенерировано заявок: {requests_count}")
    
    # 3. Строим все возможные пути
    print("\n" + "=" * 60)
    print("ПОСТРОЕНИЕ ПУТЕЙ")
    print("=" * 60)
    
    registry.build_all_paths(max_depth=50)
    
    # 4. Статистика
    print("\n" + "=" * 60)
    print("СТАТИСТИКА")
    print("=" * 60)
    
    stats = registry.get_statistics()
    print(f"Источников: {stats['total_sources']}")
    print(f"Потребителей: {stats['total_consumers']}")
    print(f"Заявок всего: {stats['total_requests']}")
    print(f"Заявок с путями: {stats['requests_with_paths']}")
    print(f"Заявок без путей: {stats['requests_without_paths']}")
    
    # 5. Выводим сводку по всем путям
    registry.print_all_paths_summary(max_per_request=3)
    
    # 6. Пример детального вывода для одной заявки (для отладки)
    if registry.requests:
        sample_request = next((req for req in registry.requests if req.paths), None)
        if sample_request:
            print("\n" + "=" * 60)
            print("ПРИМЕР ДЕТАЛЬНОГО ВЫВОДА ПУТЕЙ ДЛЯ ЗАЯВКИ")
            print("=" * 60)
            registry.print_request_paths(sample_request, max_display=10)
    
    # 7. Визуализация графа (опционально)
    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ")
    print("=" * 60)
    
    view = GraphView(graph)
    view.draw_pyvis(filename=f"{GENERETED_FOLDER}/{GRAPH_FILE}")
    
    print("\n✓ Готово!")


if __name__ == "__main__":
    main()