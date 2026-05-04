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
from graph import GraphView
from solver import FlowsCreator, Solver

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
        demand_weight=train_cfg.loss.demand_weight
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

    # Графики
    trainer.plot_loss_curves(
        filename=f"{train_cfg.paths.generated_folder}/loss_curves.png"
    )
    trainer.plot_loss_components(
        filename=f"{train_cfg.paths.generated_folder}/loss_components.png"
    )

    os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)

    # Сохранение модели
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_dim': extractor.feature_dim,
        'output_shape': extractor.get_output_shape(),
        'path_mask': path_mask
    }, f"{train_cfg.paths.generated_folder}/{train_cfg.paths.model_save_name}")

def run_prediction(graph, registry, run_cfg, train_cfg):
    with open(f"settings/{run_cfg.flows_file}") as f:
        base_flows = json.load(f)

    extractor = FeatureExtractor(graph, registry)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        # Подготовка данных для визуализации
        edge_flows = results['edge_flows']
        edge_utils = results['edge_utilization']
        
        edge_loads = {}
        for i, edge in enumerate(extractor.edges):
            edge_loads[edge] = (float(edge_flows[i]), float(edge_utils[i]))
        
        # Направленные потоки из path_flows
        directed_flows = {}
        for s_idx in range(extractor.S):
            for c_idx in range(extractor.C):
                for p_idx in range(extractor.max_paths):
                    flow = results['path_flows'][s_idx, c_idx, p_idx]
                    if flow > 0:
                        request = None
                        for req in registry.requests:
                            if (extractor.source_to_idx.get(req.source.name) == s_idx and 
                                extractor.consumer_to_idx.get(req.consumer.name) == c_idx):
                                request = req
                                break
                        if request and p_idx < len(request.paths):
                            path = request.paths[p_idx]
                            current = request.source
                            for edge in path:
                                next_node = edge.nodes[1] if edge.nodes[0] == current else edge.nodes[0]
                                key = (current.name, next_node.name)
                                directed_flows[key] = directed_flows.get(key, 0.0) + float(flow)
                                current = next_node
        
        # Создаём ОДИН файл визуализации
        view = GraphView(graph)
        os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)
        view.draw_with_directed_flows(
            edge_loads, 
            directed_flows, 
            filename=f"{train_cfg.paths.generated_folder}/ml_prediction.html",
            title="ML-предсказание потоков",
            total_demanded=results['demanded'],
            total_delivered=results['total_delivered'],
            flows_data=base_flows  # ← добавить
        )

    # Сохранение результатов
    requests = _build_requests_from_ml(results, extractor, registry, base_flows)
    edges = _build_edges_from_ml(results, extractor)
    stats = {
        'total_demanded': results['demanded'],
        'total_delivered': results['total_delivered'],
        'total_shortage': results['demanded'] - results['total_delivered'],
        'delivery_ratio': (results['total_delivered'] / results['demanded'] * 100) if results['demanded'] > 0 else 0
    }
    save_flow_results(
        stats, requests, edges,
        source_type='ML',
        filename=f"{train_cfg.paths.generated_folder}/ml_results.json"
    )

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

    # Сохранение результатов
    requests = _build_requests_from_solver(solver)
    edges = _build_edges_from_solver(solver)
    stats = {
        'total_demanded': delivery['total_requested'],
        'total_delivered': delivery['total_delivered'],
        'total_shortage': delivery['total_shortage'],
        'delivery_ratio': delivery['delivery_ratio'] * 100
    }
    save_flow_results(
        stats, requests, edges,
        source_type='Solver',
        filename=f"{train_cfg.paths.generated_folder}/solver_results.json"
    )

    # Сравнение с ML-приближением (если использовалось)
    if run_cfg.use_ml_initial_guess:
        print("Сравнение с ML-приближением:")
        print(f"  Улучшение недопоставки: {result['total_shortage']:.2f} кВт (солвер уточнил)")

    os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)
    # Визуализация
        # Визуализация (в конце run_solver_pipeline)
    if run_cfg.visualize_flows:
        os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)
        print("Визуализация решения:")
        view = GraphView(graph)
        edge_loads = solver.get_edge_loads()
        directed_flows = solver.get_directed_edge_flows()
        
        output_path = f"{train_cfg.paths.generated_folder}/solution_graph.html"
        view.draw_with_directed_flows(
            edge_loads, 
            directed_flows, 
            filename=output_path,
            title="Результат точного расчёта (солвер)",
            total_demanded=delivery['total_requested'],
            total_delivered=delivery['total_delivered'],
            flows_data=base_flows  # ← добавить
        )
    
    # График обучения солвера
    solver.plot_training_history(
        filename=f"{train_cfg.paths.generated_folder}/solver_history.png"
    )

    return result, solver

def _build_requests_from_ml(results, extractor, registry, flows):
    """Преобразует ML-результаты в список заявок."""
    requests_list = []
    for s_name, consumers in flows.items():
        if s_name not in extractor.source_to_idx:
            continue
        s_idx = extractor.source_to_idx[s_name]
        for c_name, demand in consumers.items():
            if c_name not in extractor.consumer_to_idx:
                continue
            c_idx = extractor.consumer_to_idx[c_name]
            
            request = None
            for req in registry.requests:
                if req.source.name == s_name and req.consumer.name == c_name:
                    request = req
                    break
            
            if request and s_idx < results['path_flows'].shape[0] and c_idx < results['path_flows'].shape[1]:
                total_flow = 0.0
                paths_info = []
                for p_idx, path in enumerate(request.paths):
                    if p_idx < results['path_flows'].shape[2]:
                        flow = float(results['path_flows'][s_idx, c_idx, p_idx])
                        if flow > 0.001:
                            route_names = [request.source.name]
                            current = request.source
                            for edge in path:
                                next_node = edge.nodes[1] if edge.nodes[0] == current else edge.nodes[0]
                                route_names.append(next_node.name)
                                current = next_node
                            paths_info.append({
                                'route': ' → '.join(route_names),
                                'flow': round(flow, 2)
                            })
                            total_flow += flow
                requests_list.append({
                    'source': s_name,
                    'consumer': c_name,
                    'demanded': round(demand, 2),
                    'delivered': round(total_flow, 2),
                    'shortage': round(demand - total_flow, 2),
                    'paths': paths_info
                })
    return requests_list


def _build_requests_from_solver(solver):
    """Преобразует результаты солвера в список заявок."""
    requests_list = []
    for inst in solver.instances:
        total_flow = 0.0
        paths_info = []
        for path in inst.get_paths():
            flow = inst.path_flows.get(inst._path_to_key(path), 0.0)
            if flow > 0.001:
                route_names = [inst.request.source.name]
                current = inst.request.source
                for edge in path:
                    next_node = edge.nodes[1] if edge.nodes[0] == current else edge.nodes[0]
                    route_names.append(next_node.name)
                    current = next_node
                paths_info.append({
                    'route': ' → '.join(route_names),
                    'flow': round(flow, 2)
                })
                total_flow += flow
        requests_list.append({
            'source': inst.request.source.name,
            'consumer': inst.request.consumer.name,
            'demanded': round(inst.target_amount, 2),
            'delivered': round(total_flow, 2),
            'shortage': round(inst.target_amount - total_flow, 2),
            'paths': paths_info
        })
    return requests_list


def _build_edges_from_ml(results, extractor):
    """Преобразует ML-результаты в список рёбер."""
    edges_list = []
    edge_flows = results.get('edge_flows', [])
    edge_utils = results.get('edge_utilization', [])
    for i, edge in enumerate(extractor.edges):
        cap = edge.capacity if edge.capacity != float('inf') else float('inf')
        flow = float(edge_flows[i]) if i < len(edge_flows) else 0.0
        util = float(edge_utils[i]) * 100 if i < len(edge_utils) else 0.0
        edges_list.append({
            'edge': f"{edge.nodes[0].name} ↔ {edge.nodes[1].name}",
            'capacity': 'inf' if cap == float('inf') else round(cap, 2),
            'flow': round(flow, 2),
            'utilization': round(util, 1)
        })
    return edges_list


def _build_edges_from_solver(solver):
    """Преобразует результаты солвера в список рёбер."""
    edges_list = []
    edge_loads = solver.get_edge_loads()
    for edge, (flow, ratio) in edge_loads.items():
        cap = edge.capacity if edge.capacity != float('inf') else float('inf')
        edges_list.append({
            'edge': f"{edge.nodes[0].name} ↔ {edge.nodes[1].name}",
            'capacity': 'inf' if cap == float('inf') else round(cap, 2),
            'flow': round(flow, 2),
            'utilization': round(ratio * 100, 1)
        })
    return edges_list


def save_flow_results(stats, requests, edges, source_type, filename):
    """
    Сохраняет результаты потокораспределения в JSON.
    """
    def convert(obj):
        """Рекурсивно преобразует numpy-типы в Python-типы."""
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    report = convert({
        'source': source_type,
        'summary': {
            'total_demanded': round(float(stats['total_demanded']), 2),
            'total_delivered': round(float(stats['total_delivered']), 2),
            'total_shortage': round(float(stats.get('total_shortage', 0)), 2),
            'delivery_ratio': round(float(stats['delivery_ratio']), 1)
        },
        'requests': requests,
        'edges': edges
    })
    
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✓ Результаты ({source_type}) сохранены в {filename}")
