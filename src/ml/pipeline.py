import os
import torch
import json
import numpy as np

from graph import GraphView

from .feature_extractor import FeatureExtractor
from .model import PathWeightNetwork
from .loss import EdgeFlowCalculator, PowerFlowLoss
from .inference import FlowPredictor
from .training import ModelTrainer
from .data_generator import DataGenerator
from graph import Graph, GraphView, RequestRegistry
from solver import FlowsCreator, Solver, FlowInstance
from .visualization import FlowVisualizer

def print_results(results, extractor):
    print(f"Заявлено: {results['demanded']:.1f} кВт")
    print(f"Доставлено: {results['total_delivered']:.1f} кВт")
    if results['demanded'] > 0:
        print(f"Процент доставки: {100 * results['total_delivered'] / results['demanded']:.1f}%")
    
    # Средняя доля потерь (вес фиктивного пути)
    if 'loss_weights' in results:
        mean_loss = results['loss_weights'].mean()
        print(f"Средняя доля потерь (фиктивный путь): {mean_loss:.3f}")

    edge_utils = results['edge_utilization']
    overloaded = np.where(edge_utils > 0.95)[0]
    high_load = np.where((edge_utils > 0.7) & (edge_utils <= 0.95))[0]

    print(f"\nАнализ загрузки рёбер:")
    print(f"  - Перегружено (>95%): {len(overloaded)}")
    print(f"  - Высокая загрузка (70-95%): {len(high_load)}")

    if len(overloaded) > 0:
        print(f"\n !! Перегруженные рёбра:")
        for idx in overloaded:
            edge = extractor.edges[idx]
            util = edge_utils[idx] * 100
            cap = extractor.get_edge_capacities()[idx]
            flow = results['edge_flows'][idx]
            if not np.isfinite(cap):
                print(f"  - {edge.nodes[0].name} ↔ {edge.nodes[1].name}: "
                      f"{flow:.1f} / ∞ кВт ({util:.1f}%)")
            else:
                print(f"  - {edge.nodes[0].name} ↔ {edge.nodes[1].name}: "
                      f"{flow:.1f} / {cap:.1f} кВт ({util:.1f}%)")

def run_training(graph, registry, run_cfg, train_cfg):
    # Загрузка заявок
    with open(f"settings/{run_cfg.flows_file}") as f:
        base_flows = json.load(f)

    extractor = FeatureExtractor(graph, registry)
    path_mask = extractor.create_path_mask()

    # Генерация данных
    generator = DataGenerator(
        feature_dim=extractor.feature_dim,
        sources=[s.name for s in extractor.sources],
        consumers=[c.name for c in extractor.consumers],
        E=extractor.E
    )
    raw_features, _, scenarios = generator.generate_samples(
        num_samples=train_cfg.training.num_samples_per_level,
        sparsity_levels=train_cfg.training.sparsity_levels,
        demand_scale_factors=train_cfg.training.demand_scale_factors
    )
    train_features, capacity_masks = extractor.normalize_features(raw_features)

    # Матрицы заявок
    S, C = extractor.S, extractor.C
    demands = np.zeros((len(scenarios), S, C), dtype=np.float32)
    for i, flows in enumerate(scenarios):
        for s_name, consumers in flows.items():
            s_idx = extractor.source_to_idx.get(s_name)
            if s_idx is None: continue
            for c_name, d in consumers.items():
                c_idx = extractor.consumer_to_idx.get(c_name)
                if c_idx is None: continue
                demands[i, s_idx, c_idx] = d

    split = int(0.8 * len(train_features))
    tv, tm = train_features[:split], capacity_masks[:split]
    vv, vm = train_features[split:], capacity_masks[split:]
    td, vd = demands[:split], demands[split:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PathWeightNetwork(
        input_dim=extractor.feature_dim,
        output_shape=extractor.get_output_shape(),
        hidden_dims=tuple(train_cfg.model.hidden_dims),
        dropout_rate=train_cfg.model.dropout_rate
    )
    model.set_path_mask(path_mask)

    edge_calc = EdgeFlowCalculator(registry, extractor)
    loss_fn = PowerFlowLoss(
        capacity_weight=train_cfg.loss.capacity_weight,
        demand_weight=train_cfg.loss.demand_weight,
        excess_weight=train_cfg.loss.excess_weight
    )
    trainer = ModelTrainer(model, extractor, edge_calc, loss_fn, device)
    trainer.configure_optimizer(lr=train_cfg.training.learning_rate)

    history = trainer.train(
        train_features=tv, train_demands=td, train_capacity_masks=tm,
        val_features=vv, val_demands=vd, val_capacity_masks=vm,
        epochs=train_cfg.training.epochs,
        batch_size=train_cfg.training.batch_size,
        early_stopping_patience=train_cfg.training.early_stopping_patience,
        min_delta=train_cfg.get('training.min_delta', 1e-6)
    )

    os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)

    # Сохранение модели
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_dim': extractor.feature_dim,
        'output_shape': extractor.get_output_shape(),
        'path_mask': path_mask
    }, f"{train_cfg.paths.generated_folder}/{train_cfg.paths.model_save_name}")

    # Тестирование на реальных данных
    predictor = FlowPredictor(model, extractor, edge_calc, device)
    raw_real = extractor.build_raw_features(base_flows)
    real_feat, real_mask = extractor.normalize_features(raw_real)
    results = predictor.predict_with_normalized(real_feat, base_flows, real_mask)

    print_results(results, extractor)
    if train_cfg.visualization.flows:
        fv = FlowVisualizer(graph, extractor, registry, train_cfg.paths.generated_folder)
        fv.create_flow_html(results, base_flows, 'flow_base.html', 'Базовый сценарий')

def run_prediction(graph, registry, run_cfg, train_cfg):
    with open(f"settings/{run_cfg.flows_file}") as f:
        base_flows = json.load(f)

    extractor = FeatureExtractor(graph, registry)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f" Device is {device}")
    checkpoint = torch.load(run_cfg.model_path, map_location=device, weights_only=False)
    model = PathWeightNetwork(
        input_dim=checkpoint['feature_dim'],
        output_shape=checkpoint['output_shape'],
        hidden_dims=tuple(train_cfg.model.hidden_dims),
        dropout_rate=train_cfg.model.dropout_rate
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.set_path_mask(checkpoint['path_mask'])
    model.to(device).eval()

    edge_calc = EdgeFlowCalculator(registry, extractor)
    predictor = FlowPredictor(model, extractor, edge_calc, device)
    raw_real = extractor.build_raw_features(base_flows)
    real_feat, real_mask = extractor.normalize_features(raw_real)
    results = predictor.predict_with_normalized(real_feat, base_flows, real_mask)

    print_results(results, extractor)
    if run_cfg.visualize_flows:
        fv = FlowVisualizer(graph, extractor, registry, train_cfg.paths.generated_folder)
        fv.create_flow_html(results, base_flows, 'flow_base.html', 'Базовый сценарий')

def run_solver_pipeline(graph, registry, run_cfg, train_cfg):
    """Запуск солвера с опциональным ML-начальным приближением"""
    
    # Загружаем заявки
    with open(f"settings/{run_cfg.flows_file}") as f:
        base_flows = json.load(f)

    extractor = FeatureExtractor(graph, registry)
    creator = FlowsCreator(graph, registry)
    instances = creator.create_from_file(f"settings/{run_cfg.flows_file}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Инициализация начальных потоков
    if run_cfg.get('use_ml_initial_guess', False):
        print("\nЗагрузка ML модели для начального приближения")
        checkpoint = torch.load(run_cfg.model_path, map_location=device, weights_only=False)
        model = PathWeightNetwork(
            input_dim=checkpoint['feature_dim'],
            output_shape=checkpoint['output_shape'],
            hidden_dims=tuple(train_cfg.model.hidden_dims),
            dropout_rate=train_cfg.model.dropout_rate
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.set_path_mask(checkpoint['path_mask'])
        model.to(device).eval()

        # Получаем нормализованные признаки реальных данных
        raw_real = extractor.build_raw_features(base_flows)
        real_feat, real_mask = extractor.normalize_features(raw_real)
        features_tensor = torch.from_numpy(real_feat).unsqueeze(0).float().to(device)

        with torch.no_grad():
            weights = model(features_tensor)[0].cpu().numpy()  # (S, C, max_paths)

        # Используем только реальные пути (без фиктивного)
        real_weights = weights[:, :, :extractor.max_paths]

        # Переносим веса в FlowInstance
        for inst in instances:
            s_name = inst.request.source.name
            c_name = inst.request.consumer.name
            
            if s_name not in extractor.source_to_idx or c_name not in extractor.consumer_to_idx:
                continue
            
            s_idx = extractor.source_to_idx[s_name]
            c_idx = extractor.consumer_to_idx[c_name]
            w = real_weights[s_idx, c_idx, :]
            paths = inst.get_paths()
            inst.path_flows.clear()
            
            for i, path in enumerate(paths):
                key = inst._path_to_key(path)
                if i < len(w):
                    inst.path_flows[key] = float(w[i]) * inst.target_amount
                else:
                    inst.path_flows[key] = 0.0
        print("✓ Начальное приближение от ML установлено.")
    else:
        print("!! Используется равномерное начальное распределение !!")
        
        for inst in instances:
            inst.set_uniform_flow()
        
        total_uniform = sum(inst.get_total_flow() for inst in instances)
        total_target = sum(inst.target_amount for inst in instances)
        print(f" Равномерное распределение: {total_uniform:.2f} / {total_target:.2f} кВт")

    # Запуск солвера
    print(" Запуск градиентного спуска: ")
    
    solver_cfg = train_cfg.solver
    solver = Solver(
        graph,
        learning_rate=solver_cfg.learning_rate,
        max_iter=solver_cfg.max_iter,
        epsilon=solver_cfg.epsilon,
        gradient_epsilon_rel=solver_cfg.gradient_epsilon_rel,
        capacity_weight=solver_cfg.capacity_weight,
        demand_weight=solver_cfg.demand_weight,
        excess_weight=solver_cfg.excess_weight,
        early_stopping_patience=solver_cfg.early_stopping_patience,
        verbose=solver_cfg.verbose
    )
    solver.set_instances(instances)
    result = solver.optimize()

    if not result['success']:
        print(" !!! Ошибка оптимизации:", result.get('message'))
        return None

    print(" \n Результаты оптимизации:")
    print(f"  Итераций: {result['iterations']}")
    print(f"  Финальный loss: {result['final_loss']:.2f} кВт")
    print(f"  Недопоставка: {result['total_shortage']:.2f} кВт")
    print(f"  Превышение capacity: {result['capacity_violation']:.2f} кВт")

    # Отчёт о доставке
    print(" \n Отчет о доставке энергии: ")
    
    delivery = solver.get_delivery_report()
    print(f"Всего заявлено: {delivery['total_requested']:.2f} кВт")
    print(f"Доставлено: {delivery['total_delivered']:.2f} кВт")
    print(f"Недопоставлено: {delivery['total_shortage']:.2f} кВт "
          f"({delivery['total_shortage']/delivery['total_requested']*100:.2f}%)")
    
    print("\nДетали по заявкам (первые 10):")
    for i, item in enumerate(delivery['items'][:10]):
        status = "+" if item['shortage'] < 0.01 else f"- -{item['shortage']:.1f}"
        print(f"  {i+1:2d}. {item['source']} → {item['consumer']}: "
              f"{item['delivered']:.1f} / {item['requested']:.1f} кВт {status}")
    if len(delivery['items']) > 10:
        print(f"  ... и ещё {len(delivery['items'])-10} заявок")

    # Отчёт о нарушениях пропускной способности
    print(" !! Нарушения пропускной способности: ")
    
    violations = solver.get_edge_violations()
    if violations:
        print(f" Рёбер с превышением: {len(violations)}")
        for v in violations[:10]:
            print(f"  {v['edge']}: {v['actual_flow']:.2f} / {v['capacity']:.2f} кВт "
                  f"(+{v['excess']:.2f}, {v['excess_pct']:.1f}%)")
        if len(violations) > 10:
            print(f"  ... и ещё {len(violations)-10}")
    else:
        print(" Нет рёбер с превышением пропускной способности")

    # Сравнение с ML-приближением (если использовалось)
    if run_cfg.use_ml_initial_guess:
        print("Сравнение с ML-приближением:")
        print(f"  Улучшение недопоставки: {result['total_shortage']:.2f} кВт (солвер уточнил)")

    os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)
    # Визуализация
    if run_cfg.visualize_flows:
        os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)
        print("Визуализация решения:")
        view = GraphView(graph)
        edge_loads = solver.get_edge_loads()
        directed_flows = solver.get_directed_edge_flows()
        
        output_path = f"{train_cfg.paths.generated_folder}/solution_graph.html"
        view.draw_with_directed_flows(edge_loads, directed_flows, filename=output_path)
    
    # График обучения солвера
    solver.plot_training_history(
        filename=f"{train_cfg.paths.generated_folder}/solver_history.png"
    )

    return result, solver
