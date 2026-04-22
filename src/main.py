"""Основной модуль для работы с графом электрической сети «Альфа»."""

import json
from graph import Graph, GraphView, RequestRegistry
from solver import FlowsCreator, Solver

SETTING_FOLDER: str = 'settings'
EDGES_FILE: str = 'edges.json'
FLOWS_FILE: str = 'flows.json'
GENERETED_FOLDER: str = 'genereted'
GRAPH_FILE: str = 'graph.html'
SOLUTION_GRAPH_FILE: str = 'solution_graph.html'


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
    print("ГЕНЕРАЦИЯ ЗАЯВОК И ПОСТРОЕНИЕ ПУТЕЙ")
    print("=" * 60)
    
    registry = RequestRegistry(graph)
    requests_count = registry.generate_all_requests()
    print(f"✓ Сгенерировано заявок: {requests_count}")
    registry.build_all_paths()
    
    # 3. Загружаем целевые потоки из flows.json
    print("\n" + "=" * 60)
    print("ЗАГРУЗКА ЦЕЛЕВЫХ ПОТОКОВ")
    print("=" * 60)
    creator = FlowsCreator(graph, registry)
    instances = creator.create_from_file(f"{SETTING_FOLDER}/{FLOWS_FILE}")
    print(f"✓ Загружено ненулевых потоков: {len(instances)}")
    total_target = sum(inst.target_amount for inst in instances)
    print(f"✓ Суммарная целевая мощность: {total_target:.2f} кВт")
    
    # 4. Решение задачи градиентным спуском
    print("\n" + "=" * 60)
    print("ОПТИМИЗАЦИЯ РАСПРЕДЕЛЕНИЯ ПОТОКОВ")
    print("=" * 60)
    solver = Solver(graph, learning_rate=1e-3, max_iter=10000, epsilon=1e-2, verbose=True)
    solver.set_instances(instances)
    result = solver.optimize()
    
    if result['success']:
        print(f"\n✓ Оптимизация завершена:")
        print(f"  Итераций: {result['iterations']}")
        print(f"  Суммарная недопоставка: {result['total_shortage']:.2f} кВт")
        print(f"  Суммарное превышение пропускной способности: {result['capacity_violation']:.2f} кВт")
    else:
        print("Ошибка оптимизации:", result.get('message'))
        return
    
    # 5. Отчёт о доставке
    print("\n" + "=" * 60)
    print("ОТЧЁТ О ДОСТАВКЕ ЭНЕРГИИ")
    print("=" * 60)
    delivery = solver.get_delivery_report()
    print(f"Всего заявлено: {delivery['total_requested']:.2f} кВт")
    print(f"Доставлено: {delivery['total_delivered']:.2f} кВт")
    print(f"Недопоставлено: {delivery['total_shortage']:.2f} кВт ({delivery['total_shortage']/delivery['total_requested']*100:.2f}%)")
    
    print("\nДетали по заявкам (первые 10):")
    for item in delivery['items'][:10]:
        print(f"  {item['source']} -> {item['consumer']}: "
              f"заявлено {item['requested']:.2f}, доставлено {item['delivered']:.2f}, "
              f"нехватка {item['shortage']:.2f} ({item['shortage_pct']:.1f}%)")
    if len(delivery['items']) > 10:
        print(f"  ... и ещё {len(delivery['items'])-10}")
    
    # 6. Отчёт о нарушениях пропускной способности
    print("\n" + "=" * 60)
    print("НАРУШЕНИЯ ПРОПУСКНОЙ СПОСОБНОСТИ")
    print("=" * 60)
    violations = solver.get_edge_violations()
    if violations:
        print(f"Рёбер с превышением: {len(violations)}")
        for v in violations[:10]:
            print(f"  {v['edge']}: capacity={v['capacity']:.2f}, flow={v['actual_flow']:.2f}, "
                  f"excess={v['excess']:.2f} ({v['excess_pct']:.1f}%)")
        if len(violations) > 10:
            print(f"  ... и ещё {len(violations)-10}")
    else:
        print("Нет рёбер с превышением пропускной способности.")
    
    # 7. Визуализация результата
    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТА")
    print("=" * 60)
    view = GraphView(graph)
    edge_loads = solver.get_edge_loads()
    view.draw_with_loads(edge_loads, filename=f"{GENERETED_FOLDER}/{SOLUTION_GRAPH_FILE}")

    # В конце main() добавьте:
    solver.plot_training_history(filename=f"{GENERETED_FOLDER}/training_history.png")
    
    print("\n✓ Готово!")


if __name__ == "__main__":
    main()