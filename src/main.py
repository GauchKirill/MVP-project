import torch
import json
from graph import Graph, GraphView, RequestRegistry
from ml import (
    FeatureExtractor, DataGenerator,
    PathWeightNetwork, PowerFlowLoss, EdgeFlowCalculator,
    ModelTrainer, FlowPredictor
)
from ml.data_generator import DataVisualizer

from ml.visualization import TrainingVisualizer, FlowVisualizer
import numpy as np

SETTING_FOLDER: str = 'settings'
EDGES_FILE: str = 'edges.json'
GENERETED_FOLDER: str = 'genereted'
GRAPH_FILE: str = 'graph.html'

# === НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ ===
TRAINING_CONFIG = {
    'num_samples_per_level': 1000,  # сэмплов на каждый уровень разреженности
    'sparsity_levels': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # уровни разреженности
    'batch_size': 128,
    'epochs': 200,
    'learning_rate': 1e-4,
    'early_stopping_patience': 10,
}

# === ФЛАГИ ===
VISUALIZE_TRAINING = True       # строить ли графики обучения
VISUALIZE_FLOWS = True          # создавать ли HTML визуализацию
SAVE_REPORT = True              # сохранять ли JSON отчёт
VISUALIZE_DATA = True           # визуализировать ли сгенерированные данные

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
    print(f"  - Сэмплов на уровень: {config['num_samples_per_level']}")
    print(f"  - Уровни разреженности: {config['sparsity_levels']}")
    print(f"  - Всего уровней: {len(config['sparsity_levels'])}")
    print(f"  - Батчей: {config['batch_size']}")
    print(f"  - Эпох (макс): {config['epochs']}")
    print(f"  - Learning rate: {config['learning_rate']}")
    
    # Загружаем flows.json
    with open(f"{SETTING_FOLDER}/flows.json", 'r', encoding='utf-8') as f:
        base_flows = json.load(f)
    
    # Создаём экстрактор признаков
    extractor = FeatureExtractor(graph, registry)
    path_mask = extractor.create_path_mask()
    print(f"✓ Размерность признаков: {extractor.feature_dim}")
    print(f"✓ Форма выхода: {extractor.get_output_shape()}")
    print(f"✓ Максимальное число путей: {extractor.max_paths}")
    
    # Устанавливаем маску путей
    path_mask = extractor.create_path_mask()
    
    # Генерируем обучающие данные
    print("\nГенерация обучающих данных...")
    generator = DataGenerator(
        feature_dim=extractor.feature_dim,
        sources=[s.name for s in extractor.sources],
        consumers=[c.name for c in extractor.consumers],
        E=extractor.E
    )

    # Получаем сырые признаки (с inf) и сценарии
    raw_features, demands_matrix, scenarios = generator.generate_samples(
        num_samples=config['num_samples_per_level'],
        sparsity_levels=config['sparsity_levels']
    )

    # Нормализуем и получаем маски
    train_features, capacity_masks = extractor.normalize_features(raw_features)

    print(f"✓ Сгенерировано {len(scenarios)} сценариев")
    print(f"✓ Размер матрицы признаков: {train_features.shape}")
    print(f"  - Мин значение: {train_features.min():.6e}")
    print(f"  - Макс значение: {train_features.max():.4f}")
    print(f"  - Среднее: {train_features.mean():.6e}")

    # Создаём матрицы заявок ИЗ СЦЕНАРИЕВ (не из features!)
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
    split_idx = int(0.8 * len(train_features))
    val_features = train_features[split_idx:]
    val_demands = demands_matrix[split_idx:]
    val_masks = capacity_masks[split_idx:]
    train_features = train_features[:split_idx]
    train_demands = demands_matrix[:split_idx]
    train_masks = capacity_masks[:split_idx]

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
    loss_fn = PowerFlowLoss(
        capacity_weight=10.0,
        demand_weight=1.0
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
        train_capacity_masks=train_masks,
        val_features=val_features,
        val_demands=val_demands,
        val_capacity_masks=val_masks,
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

    # Тестирование на реальных данных
    predictor = FlowPredictor(model, extractor, edge_calculator, device)
    raw_real = extractor.build_raw_features(base_flows)
    real_features, real_mask = extractor.normalize_features(raw_real)
    results = predictor.predict_with_normalized(real_features, base_flows, real_mask)

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
            if ~np.isfinite(cap):
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
    
    return predictor, results

if __name__ == "__main__":
    main()