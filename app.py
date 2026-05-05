"""
Веб-интерфейс для системы оптимизации распределения потоков.
Запуск: streamlit run app.py
"""

import streamlit as st
import json
import os
import sys
import io
import time
import re
import numpy as np
import pandas as pd
from PIL import Image
import threading
import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config_parser import ConfigParser
from graph import Graph, GraphView, RequestRegistry
from ml.pipeline import run_training, run_prediction, run_solver_pipeline

st.set_page_config(
    page_title="Транснефть: Оптимизация потоков",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 1rem; }
    .stButton > button { width: 100%; height: 3rem; font-size: 1.2rem; font-weight: bold; }
    .training-log { background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.9rem; max-height: 400px; overflow-y: auto; }
    .epoch-badge { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0; }
    .solver-badge { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# Глобальный флаг для остановки предыдущего процесса
if 'active_thread' not in st.session_state:
    st.session_state.active_thread = None
if 'stop_flag' not in st.session_state:
    st.session_state.stop_flag = threading.Event()


def load_graph_and_registry(edges_path: str) -> tuple:
    graph = Graph()
    with open(edges_path, 'r', encoding='utf-8') as f:
        edges_data = json.load(f)
    for item in edges_data:
        n1, n2 = item['nodes']
        cap = item['capacity']
        cap = float('inf') if cap == 'inf' else float(cap)
        graph.add_edge(n1, n2, cap)
    registry = RequestRegistry(graph)
    registry.generate_all_requests()
    registry.build_all_paths()
    return graph, registry


def display_metrics(demanded, delivered, edge_utils):
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Заявлено", f"{demanded:,.1f} кВт")
    with col2:
        ratio = delivered / demanded * 100 if demanded > 0 else 0
        st.metric("Доставлено", f"{delivered:,.1f} кВт", delta=f"{ratio:.1f}%")
    with col3:
        overloaded = int((edge_utils > 0.95).sum())
        st.metric("Перегружено рёбер", overloaded, delta="!" if overloaded > 0 else "OK")
    with col4:
        high_load = int(((edge_utils > 0.7) & (edge_utils <= 0.95)).sum())
        st.metric("Высокая загрузка", high_load)


def stop_previous_process():
    """Останавливает предыдущий активный процесс."""
    if st.session_state.active_thread is not None:
        st.session_state.stop_flag.set()
        st.session_state.active_thread.join(timeout=2.0)
        st.session_state.stop_flag.clear()
        st.session_state.active_thread = None


def main():
    st.markdown('<p class="main-header">⚡ Система оптимизации распределения электрической энергии</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Электрическая сеть "Альфа" | Транснефть</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.header("Режим работы")
        st.markdown("---")
        mode = st.radio("Выберите режим:", ["ML-обучение", "ML-предсказание", "Точное решение (ML + солвер)"])
        st.markdown("---")
        st.header("Данные")
        use_default = st.checkbox("Использовать стандартные данные", value=True)
        
        if use_default:
            edges_path = "settings/edges.json"
            flows_path = "settings/flows.json"
            st.success("Стандартные данные")
        else:
            edges_file = st.file_uploader("Ребра графа (JSON)", type=["json"])
            flows_file = st.file_uploader("Заявки (JSON)", type=["json"])
            if edges_file and flows_file:
                os.makedirs("uploads", exist_ok=True)
                edges_path = f"uploads/edges_{int(time.time())}.json"
                flows_path = f"uploads/flows_{int(time.time())}.json"
                with open(edges_path, 'wb') as f: f.write(edges_file.getbuffer())
                with open(flows_path, 'wb') as f: f.write(flows_file.getbuffer())
                st.success("Файлы загружены")
            else:
                st.warning("Загрузите оба файла")
                st.stop()
        
        # === РЕЖИМ 1: ML-обучение ===
        if mode == "ML-обучение":
            st.markdown("---")
            st.header("Параметры обучения")
            
            with st.expander("Данные", expanded=True):
                num_samples = st.number_input("Сэмплов на уровень", 100, 5000, 500, 100, key="num_samples")
                sparsity_str = st.text_input("Уровни разреженности (через запятую)", value="0.1, 0.3, 0.5, 0.7, 0.9", key="sparsity_str")
                sparsity_options = [float(x.strip()) for x in sparsity_str.split(",") if x.strip()]
                demand_str = st.text_input("Масштабы заявок (через запятую)", value="0.02, 0.04", key="demand_str")
                demand_scales = [float(x.strip()) for x in demand_str.split(",") if x.strip()]
            
            with st.expander("Обучение", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    batch_size = st.number_input("Размер батча", 16, 512, 128, 16, key="batch_size")
                    epochs = st.number_input("Количество эпох", 10, 500, 100, 10, key="epochs")
                    lr = st.number_input("Learning rate", 1e-6, 1e-1, 1e-3, format="%.6f", key="lr")
                with col2:
                    patience = st.number_input("Терпение (early stopping)", 3, 50, 10, key="patience")
                    min_delta = st.number_input("Min delta", 1e-8, 1e-2, 1e-4, format="%.6f", key="min_delta")
            
            with st.expander("Функция потерь", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1: demand_weight = st.number_input("demand_weight", 0.1, 1000.0, 10.0, key="demand_w")
                with col2: excess_weight = st.number_input("excess_weight", 0.1, 100.0, 1.5, key="excess_w")
                with col3: capacity_weight = st.number_input("capacity_weight", 0.1, 100.0, 1.8, key="capacity_w")
        
        # === РЕЖИМ 2: ML-предсказание ===
        elif mode == "ML-предсказание":
            st.markdown("---")
            st.header("Модель для предсказания")
            model_file = st.file_uploader("Загрузите файл модели (.pt)", type=["pt"])
            if model_file:
                os.makedirs("uploads", exist_ok=True)
                model_path = f"uploads/{model_file.name}"
                with open(model_path, 'wb') as f: f.write(model_file.getbuffer())
                st.success(f"Модель загружена: {model_file.name}")
            else:
                model_path = "genereted/model.pt"
                st.info(f"Используется стандартная модель: {model_path}")
        
        # === РЕЖИМ 3: Точное решение ===
        elif mode == "Точное решение (ML + солвер)":
            st.markdown("---")
            st.header("ML-модель для начального приближения")
            model_file = st.file_uploader("Загрузите файл модели (.pt)", type=["pt"])
            if model_file:
                os.makedirs("uploads", exist_ok=True)
                solver_model_path = f"uploads/{model_file.name}"
                with open(solver_model_path, 'wb') as f: f.write(model_file.getbuffer())
                st.success(f"Модель загружена: {model_file.name}")
            else:
                solver_model_path = "genereted/model.pt"
                st.info(f"Используется стандартная модель: {solver_model_path}")
            
            st.markdown("---")
            st.header("Параметры солвера")
            with st.expander("Градиентный спуск", expanded=True):
                solver_lr = st.number_input("Learning rate", 0.001, 2.0, 0.3, format="%.4f", key="solver_lr")
                solver_max_iter = st.number_input("Макс. итераций", 100, 10000, 1000, 100, key="solver_iter")
                solver_epsilon = st.number_input("Эпсилон сходимости", 1e-8, 1e-1, 1e-4, format="%.6f", key="solver_eps")
                solver_patience = st.number_input("Терпение", 5, 100, 20, key="solver_patience")
                solver_capacity_w = st.number_input("capacity_weight", 0.1, 100.0, 1.8, key="solver_cap_w")
        
        st.markdown("---")
        run_button = st.button("ЗАПУСТИТЬ", type="primary", width='stretch')
    
    if not run_button:
        st.info("Выберите режим и нажмите ЗАПУСТИТЬ")
        return
    
    # Останавливаем предыдущий процесс перед запуском нового
    stop_previous_process()
    
    with st.spinner("Загрузка графа..."):
        graph, registry = load_graph_and_registry(edges_path)
    st.success(f"Граф загружен: {len(graph.nodes)} узлов, {len(graph.edges)} ребер")
    
    # ==================== РЕЖИМ 1: ML-обучение ====================
    if mode == "ML-обучение":
        import torch
        import re
        
        st.subheader("ML-обучение")
        
        config = {
            "training": {
                "num_samples_per_level": num_samples, "sparsity_levels": sparsity_options,
                "demand_scale_factors": demand_scales, "batch_size": batch_size,
                "epochs": epochs, "learning_rate": lr, "early_stopping_patience": patience,
                "min_delta": min_delta
            },
            "model": {"hidden_dims": [512, 256, 128], "dropout_rate": 0.3},
            "loss": {"demand_weight": demand_weight, "excess_weight": excess_weight, "capacity_weight": capacity_weight},
            "paths": {"generated_folder": "genereted", "model_save_name": "model.pt"}
        }
        with open('settings/config.json', 'w', encoding='utf-8') as f: json.dump(config, f, indent=2)
        train_cfg = ConfigParser('settings/config.json')
        
        run_config = {"edges_file": os.path.basename(edges_path), "flows_file": os.path.basename(flows_path), "mode": "train", "visualize_flows": False}
        with open('settings/run_config.json', 'w', encoding='utf-8') as f: json.dump(run_config, f, indent=2)
        run_cfg = ConfigParser('settings/run_config.json')
        os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        log_buffer = io.StringIO()
        
        class TeeOutput:
            def __init__(self, buffer, original_stream):
                self.buffer = buffer
                self.original = original_stream
            def write(self, message):
                self.original.write(message)
                self.original.flush()
                self.buffer.write(message)
            def flush(self):
                self.original.flush()
        
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeOutput(log_buffer, original_stdout)
        sys.stderr = TeeOutput(log_buffer, original_stderr)
        
        try:
            status.markdown('<div class="epoch-badge"><b>Обучение запущено...</b></div>', unsafe_allow_html=True)
            
            run_training(graph, registry, run_cfg, train_cfg)
            
            progress_bar.progress(100)
            status.success("Обучение завершено!")
            
            raw_logs = log_buffer.getvalue()
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            clean_logs = ansi_escape.sub('', raw_logs)
            
            epoch_lines = []
            for line in clean_logs.split('\n'):
                if 'Обучение:' in line and 'эпоха' in line and 'train=' in line:
                    if 'эпоха,' in line:
                        line = line.split('эпоха,', 1)[-1].strip()
                    epoch_lines.append(line)
            
            unique_lines = []
            prev_line = ""
            for line in epoch_lines:
                if line != prev_line:
                    unique_lines.append(line)
                    prev_line = line
            
            clean_logs = '\n'.join(unique_lines)
            
            with st.expander("Итоги обучения по эпохам"):
                st.text_area("Логи", clean_logs, height=400)
            
            gen_folder = train_cfg.paths.generated_folder
            for f, cap in [("loss_curves.png", "Кривые обучения"), ("loss_components.png", "Компоненты функции потерь")]:
                path = f"{gen_folder}/{f}"
                if os.path.exists(path):
                    st.image(Image.open(path), caption=cap, width='stretch')
            
            st.success(f"Модель сохранена в {gen_folder}/{train_cfg.paths.model_save_name}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    
    # ==================== РЕЖИМ 2: ML-предсказание ====================
    elif mode == "ML-предсказание":
        st.subheader("ML-предсказание")
        
        config = {"model": {"hidden_dims": [512, 256, 128], "dropout_rate": 0.3}, "paths": {"generated_folder": "genereted", "model_save_name": "model.pt"}}
        with open('settings/config.json', 'w', encoding='utf-8') as f: json.dump(config, f, indent=2)
        train_cfg = ConfigParser('settings/config.json')
        
        run_config = {"edges_file": os.path.basename(edges_path), "flows_file": os.path.basename(flows_path), "mode": "predict", "model_path": model_path, "visualize_flows": True}
        with open('settings/run_config.json', 'w', encoding='utf-8') as f: json.dump(run_config, f, indent=2)
        run_cfg = ConfigParser('settings/run_config.json')
        os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)
        
        with st.spinner("Загрузка модели и предсказание..."):
            log_buffer = io.StringIO()
            
            class TeeOutput:
                def __init__(self, buffer, original_stream):
                    self.buffer = buffer
                    self.original = original_stream
                def write(self, message):
                    self.original.write(message)
                    self.original.flush()
                    self.buffer.write(message)
                def flush(self):
                    self.original.flush()
            
            original_stdout = sys.stdout
            sys.stdout = TeeOutput(log_buffer, original_stdout)
            
            try:
                run_prediction(graph, registry, run_cfg, train_cfg)
            finally:
                sys.stdout = original_stdout
        
        with st.expander("Логи предсказания"):
            st.text_area("Логи", log_buffer.getvalue(), height=300)
        
        st.markdown("---")
        st.subheader("Граф потоков")
        ml_graph_path = f"{train_cfg.paths.generated_folder}/ml_prediction.html"
        if os.path.exists(ml_graph_path):
            with open(ml_graph_path, 'r', encoding='utf-8') as f:
                st.components.v1.html(f.read(), height=700, scrolling=True)
        else:
            st.warning("Файл графа не найден")
    
    # ==================== РЕЖИМ 3: Точное решение ====================
    elif mode == "Точное решение (ML + солвер)":
        import re
        
        st.subheader("Точное решение (ML + солвер)")
        
        config = {
            "model": {"hidden_dims": [512, 256, 128], "dropout_rate": 0.3},
            "solver": {
                "learning_rate": solver_lr, "max_iter": solver_max_iter,
                "epsilon": solver_epsilon, "early_stopping_patience": solver_patience,
                "gradient_epsilon_rel": 0.01, "capacity_weight": solver_capacity_w,
                "verbose": True
            },
            "paths": {"generated_folder": "genereted", "model_save_name": "model.pt"}
        }
        with open('settings/config.json', 'w', encoding='utf-8') as f: json.dump(config, f, indent=2)
        train_cfg = ConfigParser('settings/config.json')
        
        run_config = {"edges_file": os.path.basename(edges_path), "flows_file": os.path.basename(flows_path), "mode": "solve", "use_ml_initial_guess": True, "model_path": solver_model_path, "visualize_flows": True}
        with open('settings/run_config.json', 'w', encoding='utf-8') as f: json.dump(run_config, f, indent=2)
        run_cfg = ConfigParser('settings/run_config.json')
        os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)
        
        progress_bar = st.progress(0)
        status = st.empty()
        iter_placeholder = st.empty()
        
        log_buffer = io.StringIO()
        
        class TeeOutput:
            def __init__(self, buffer, original_stream):
                self.buffer = buffer
                self.original = original_stream
            def write(self, message):
                self.original.write(message)
                self.original.flush()
                self.buffer.write(message)
            def flush(self):
                self.original.flush()
        
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeOutput(log_buffer, original_stdout)
        sys.stderr = TeeOutput(log_buffer, original_stderr)
        
        try:
            status.markdown('<div class="solver-badge"><b>Расчет запущен...</b></div>', unsafe_allow_html=True)
            progress_bar.progress(10)
            
            solver_done = threading.Event()
            solver_result = {}
            
            def solver_thread():
                r, s = run_solver_pipeline(graph, registry, run_cfg, train_cfg)
                solver_result['result'] = r
                solver_result['solver'] = s
                solver_done.set()
            
            thread = threading.Thread(target=solver_thread)
            thread.start()
            st.session_state.active_thread = thread
            
            while not solver_done.is_set():
                if st.session_state.stop_flag.is_set():
                    st.warning("Процесс остановлен пользователем.")
                    break
                time.sleep(0.5)
                current_logs = log_buffer.getvalue()
                
                iter_matches = re.findall(r'Итерация\s+(\d+):\s*loss\s*=\s*([\d.]+)', current_logs)
                if iter_matches:
                    iter_num = int(iter_matches[-1][0])
                    loss_val = float(iter_matches[-1][1])
                    pct = 10 + min(iter_num / max(solver_max_iter, 1), 1.0) * 85
                    progress_bar.progress(int(pct))
                    
                    loss_color = "#27ae60" if loss_val < 100 else "#f39c12" if loss_val < 1000 else "#e74c3c"
                    iter_placeholder.markdown(
                        f'<div class="solver-badge"><b>Итерация {iter_num} / {solver_max_iter}</b><br><small>Текущий loss: <b style="color:{loss_color};">{loss_val:.2f} кВт</b></small></div>',
                        unsafe_allow_html=True
                    )
            
            st.session_state.active_thread = None
            
            if not st.session_state.stop_flag.is_set():
                progress_bar.progress(95)
                status.success("Расчет завершен!")
                iter_placeholder.empty()
                
                raw_logs = log_buffer.getvalue()
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                clean_logs = ansi_escape.sub('', raw_logs)
                
                log_lines = []
                for line in clean_logs.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    if any(kw in line for kw in ['Итерация', 'loss', 'недопоставка', 'превышение', 'Всего заявлено', 'Доставлено', 'рёбер с превышением', 'Финальный']):
                        log_lines.append(line)
                
                with st.expander("Логи солвера"):
                    st.text_area("Логи", '\n'.join(log_lines), height=300)
                
                result = solver_result.get('result')
                solver = solver_result.get('solver')
                
                if result and solver:
                    delivery = solver.get_delivery_report()
                    
                    st.markdown("---")
                    st.subheader("Результаты точного решения")
                    edge_loads = solver.get_edge_loads()
                    edge_utils_arr = np.array([r for _, r in edge_loads.values()])
                    display_metrics(delivery['total_requested'], delivery['total_delivered'], edge_utils_arr)
                    
                    st.markdown("---")
                    st.subheader("Граф потоков")
                    output_path = f"{train_cfg.paths.generated_folder}/solution_graph.html"
                    if os.path.exists(output_path):
                        with open(output_path, 'r', encoding='utf-8') as f:
                            st.components.v1.html(f.read(), height=700, scrolling=True)
                    else:
                        st.warning("Файл графа не найден")
                    
                    st.markdown("---")
                    st.subheader("Обучение солвера")
                    history_file = f"{train_cfg.paths.generated_folder}/solver_history.png"
                    if os.path.exists(history_file):
                        st.image(Image.open(history_file), width='stretch')
                
                progress_bar.progress(100)
        finally:
            st.session_state.active_thread = None
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
