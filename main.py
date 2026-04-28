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

    graph = Graph()
    load_edges(graph, f"settings/{run_cfg.edges_file}")
    registry = RequestRegistry(graph)
    registry.generate_all_requests()
    registry.build_all_paths()

    mode = run_cfg.mode
    if mode == 'train':
        run_training(graph, registry, run_cfg, train_cfg)
    elif mode == 'predict':
        run_prediction(graph, registry, run_cfg, train_cfg)
    elif mode == 'solve':
        run_solver_pipeline(graph, registry, run_cfg, train_cfg)
    else:
        raise ValueError(f"Неизвестный режим: {mode}")

if __name__ == "__main__":
    main()