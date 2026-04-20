import torch
import json
from graph import Graph, GraphView, RequestRegistry
from ml import (
        FeatureExtractor, DataGenerator, PhysicsValidator,
        PathWeightNetwork, PowerFlowLoss, EdgeFlowCalculator,
        ModelTrainer, FlowPredictor
    )

from ml.visualization import TrainingVisualizer, FlowVisualizer
import numpy as np

SETTING_FOLDER: str = 'settings'
EDGES_FILE: str = 'edges.json'
GENERETED_FOLDER: str = 'genereted'
GRAPH_FILE: str = 'graph.html'

# === НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ ===
TRAINING_CONFIG = {
    'num_samples': 200,        # количество сценариев для обучения (уменьшено для скорости)
    'batch_size': 32,           # размер батча
    'epochs': 50,               # максимальное количество эпох
    'learning_rate': 1e-4,      # скорость обучения
    'early_stopping_patience': 15,  # терпение для ранней остановки
    'lhs_ratio': 0.3,           # доля LHS сэмплов
    'noise_std': 0.3,           # стандартное отклонение шума для вариаций
}

# === ФЛАГИ ===
VISUALIZE_TRAINING = True       # строить ли графики обучения
VISUALIZE_FLOWS = True          # создавать ли HTML визуализацию
RUN_SENSITIVITY = True          # запускать ли анализ чувствительности
SAVE_REPORT = True              # сохранять ли JSON отчёт

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
    
    # 8. Обучение нейросетевой модели (опционально)
    print("\n" + "=" * 60)
    response = input("Обучить нейросетевую модель? (y/n): ")
    if response.lower() == 'y':
        train_model(graph, registry)  # Передаём graph и registry!
    
def train_model(graph, registry, config=None):
    """Обучение нейросетевой модели."""
    if config is None:
        config = TRAINING_CONFIG
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ НЕЙРОСЕТЕВОЙ МОДЕЛИ")
    print("=" * 60)
    print(f"Конфигурация:")
    print(f"  - Обучающих сценариев: {config['num_samples']}")
    print(f"  - Батчей: {config['batch_size']}")
    print(f"  - Эпох (макс): {config['epochs']}")
    print(f"  - Learning rate: {config['learning_rate']}")
    
    # Загружаем flows.json
    with open(f"{SETTING_FOLDER}/flows.json", 'r', encoding='utf-8') as f:
        base_flows = json.load(f)
    
    # Создаём экстрактор признаков
    extractor = FeatureExtractor(graph, registry)
    print(f"✓ Размерность признаков: {extractor.feature_dim}")
    print(f"✓ Форма выхода: {extractor.get_output_shape()}")
    print(f"✓ Максимальное число путей: {extractor.max_paths}")
    
    # Устанавливаем маску путей
    path_mask = extractor.create_path_mask()
    
    # Генерируем обучающие данные
    print("\nГенерация обучающих данных...")
    generator = DataGenerator(
        base_flows=base_flows,
        feature_dim=extractor.feature_dim,
        sources=[s.name for s in extractor.sources],
        consumers=[c.name for c in extractor.consumers]
    )
    
    scenarios = generator.generate_mixed_samples(
        total_samples=config['num_samples'],
        lhs_ratio=config['lhs_ratio'],
        sparsity=0.7,
        noise_std=config['noise_std']
    )
    
    # Фильтруем физически реализуемые сценарии
    validator = PhysicsValidator(graph, extractor)
    scenarios = validator.filter_feasible(scenarios)
    print(f"✓ Сгенерировано {len(scenarios)} реализуемых сценариев")
    
    # Извлекаем признаки и заявки
    train_features = extractor.extract_batch_features(scenarios)
    
    # Создаём матрицы заявок
    S, C = extractor.S, extractor.C
    train_demands = np.zeros((len(scenarios), S, C), dtype=np.float32)
    for i, flows in enumerate(scenarios):
        for s_name, consumers in flows.items():
            if s_name in extractor.source_to_idx:
                s_idx = extractor.source_to_idx[s_name]
                for c_name, demand in consumers.items():
                    if c_name in extractor.consumer_to_idx:
                        c_idx = extractor.consumer_to_idx[c_name]
                        train_demands[i, s_idx, c_idx] = demand
    
    # Разделяем на train/val
    split_idx = int(0.8 * len(scenarios))
    val_features = train_features[split_idx:] if split_idx < len(train_features) else train_features
    val_demands = train_demands[split_idx:] if split_idx < len(train_demands) else train_demands
    train_features = train_features[:split_idx]
    train_demands = train_demands[:split_idx]
    
    print(f"✓ Train: {len(train_features)}, Val: {len(val_features)}")
    
    # Создаём модель
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Используется устройство: {device}")
    
    model = PathWeightNetwork(
        input_dim=extractor.feature_dim,
        output_shape=extractor.get_output_shape(),
        hidden_dims=(512, 256, 128),
        dropout_rate=0.3
    )
    model.set_path_mask(path_mask)
    
    # Создаём вычислитель потоков на рёбрах и функцию потерь
    edge_calculator = EdgeFlowCalculator(registry, extractor)
    # В train_model() при создании loss_fn:
    # Теперь веса имеют смысл относительной важности
    loss_fn = PowerFlowLoss(
        capacity_weight=10.0,   # Перегрузка в 10 раз важнее недопоставки
        demand_weight=1.0       # Базовая важность доставки
    )
    
    # Обучаем модель
    trainer = ModelTrainer(model, extractor, edge_calculator, loss_fn, device)
    trainer.configure_optimizer(
        lr=config['learning_rate'], 
        weight_decay=1e-5
    )
    
    print("\nНачало обучения...")
    history = trainer.train(
        train_features=train_features,
        train_demands=train_demands,
        val_features=val_features,
        val_demands=val_demands,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        early_stopping_patience=config['early_stopping_patience'],
        verbose=True
    )
    
    # Визуализация истории обучения
    if VISUALIZE_TRAINING:
        train_viz = TrainingVisualizer(save_dir=GENERETED_FOLDER)
        train_viz.plot_training_history(history, show=False)
    
    # Сохраняем модель
    model_save_path = f"{GENERETED_FOLDER}/model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_dim': extractor.feature_dim,
        'output_shape': extractor.get_output_shape(),
        'path_mask': path_mask,
        'config': config,
    }, model_save_path)
    print(f"\n✓ Модель сохранена в {model_save_path}")
    
    # Тестируем на реальных данных
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 60)
    
    predictor = FlowPredictor(model, extractor, edge_calculator, device)
    results = predictor.predict(base_flows)
    
    print(f"Заявлено: {results['demanded']:.1f} кВт")
    print(f"Доставлено: {results['total_delivered']:.1f} кВт")
    if results['demanded'] > 0:
        print(f"Процент доставки: {100 * results['total_delivered'] / results['demanded']:.1f}%")
    
    # Анализируем загрузку рёбер
    edge_utils = results['edge_utilization']
    overloaded = np.where(edge_utils > 0.95)[0]
    high_load = np.where((edge_utils > 0.7) & (edge_utils <= 0.95))[0]
    
    print(f"\nАнализ загрузки рёбер:")
    print(f"  - Перегружено (>95%): {len(overloaded)}")
    print(f"  - Высокая загрузка (70-95%): {len(high_load)}")
    
    if len(overloaded) > 0:
        print(f"\n⚠️ Перегруженные рёбра:")
        for idx in overloaded:
            edge = extractor.edges[idx]
            util = edge_utils[idx] * 100
            cap = extractor.get_edge_capacities()[idx]
            flow = results['edge_flows'][idx]
            if cap >= 1e8:
                print(f"  - {edge.nodes[0].name} ↔ {edge.nodes[1].name}: "
                    f"{flow:.1f} / ∞ кВт ({util:.1f}%)")
            else:
                print(f"  - {edge.nodes[0].name} ↔ {edge.nodes[1].name}: "
                    f"{flow:.1f} / {cap:.1f} кВт ({util:.1f}%)")
    
    # Визуализация потоков
    if VISUALIZE_FLOWS:
        flow_viz = FlowVisualizer(graph, extractor, registry, save_dir=GENERETED_FOLDER)
        
        # Визуализация для базовых заявок
        flow_viz.create_flow_html(
            results, base_flows,
            save_name='flow_base.html',
            title='Базовый сценарий (исходные заявки)'
        )
        
        # Сохраняем отчёт
        if SAVE_REPORT:
            flow_viz.save_results_report(results, base_flows, 'results_base.json')
    
    # Анализ чувствительности
    if RUN_SENSITIVITY:
        print("\n" + "=" * 60)
        print("АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ")
        print("=" * 60)
        
        flow_viz = FlowVisualizer(graph, extractor, registry, save_dir=GENERETED_FOLDER)
        sensitivity_results = flow_viz.plot_sensitivity_analysis(
            predictor,
            base_flows,
            parameter='demand',
            variation_range=(0.5, 2.0),
            num_points=15,
            save_name='sensitivity_demand.png',
            show=False
        )
        
        # Выводим ключевые точки
        print("\nКлючевые точки:")
        mults = sensitivity_results['multipliers']
        deliveries = sensitivity_results['delivery_ratios']
        overloads = sensitivity_results['overload_counts']
        
        # Находим точку, где начинаются перегрузки
        for i, (mult, ov) in enumerate(zip(mults, overloads)):
            if ov > 0:
                print(f"  - Перегрузки начинаются при множителе {mult:.2f}")
                print(f"    (заявки увеличены на {(mult-1)*100:.0f}%)")
                break
        
        # Находим точку, где доставка падает ниже 95%
        for i, (mult, deliv) in enumerate(zip(mults, deliveries)):
            if deliv < 95:
                print(f"  - Доставка падает ниже 95% при множителе {mult:.2f}")
                break
    
    print("\n✓ Обучение и тестирование завершены!")
    
    return predictor, results

if __name__ == "__main__":
    main()