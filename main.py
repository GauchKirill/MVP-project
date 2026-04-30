import os
import json
from src.config_parser import ConfigParser
from src.graph import Graph, RequestRegistry
from src.ml.pipeline import run_training, run_prediction, run_solver_pipeline

def load_edges(graph, filename):
    with open(filename, 'r') as f:
        for item in json.load(f):
            n1, n2 = item['nodes']
            cap = item['capacity']
            if cap == 'inf':
                cap = float('inf')
            graph.add_edge(n1, n2, float(cap))

def main():
    run_cfg = ConfigParser('settings/run_config.json')
    train_cfg = ConfigParser('settings/config.json')

    os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)

    graph = Graph()
    load_edges(graph, f"settings/{run_cfg.edges_file}")
    registry = RequestRegistry(graph)
    registry.generate_all_requests()
    registry.build_all_paths()

    mode = run_cfg.mode
    if mode == 'train':  # режим обучения на сгенерированных данных
        run_training(graph, registry, run_cfg, train_cfg)
    elif mode == 'predict':  # режим предсказания без переобучения (хорошее приближение для тестирования на новых данныз)
        run_prediction(graph, registry, run_cfg, train_cfg)
    elif mode == 'solve':  # режим точного решения (советуем выполнять после приближенного предсказания от ml решателя)
        run_solver_pipeline(graph, registry, run_cfg, train_cfg)
    else:
        raise ValueError(f"Неизвестный режим: {mode}")

if __name__ == "__main__":
    main()