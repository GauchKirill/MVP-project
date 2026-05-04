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
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
from PIL import Image
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config_parser import ConfigParser
from graph import Graph, GraphView, RequestRegistry
from solver import Solver, FlowsCreator
from ml.feature_extractor import FeatureExtractor

st.set_page_config(
    page_title="Транснефть: Оптимизация потоков",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .training-log {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.9rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .epoch-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .solver-badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_graph_and_registry(edges_path: str) -> tuple:
    """Загружает граф и реестр заявок."""
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
    """Отображает ключевые метрики."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Заявлено", f"{demanded:,.1f} кВт")
    with col2:
        ratio = delivered / demanded * 100 if demanded > 0 else 0
        st.metric("Доставлено", f"{delivered:,.1f} кВт", delta=f"{ratio:.1f}%")
    with col3:
        overloaded = int((edge_utils > 0.95).sum())
        st.metric("Перегружено рёбер", overloaded, delta="⚠️" if overloaded > 0 else "✓")
    with col4:
        high_load = int(((edge_utils > 0.7) & (edge_utils <= 0.95)).sum())
        st.metric("Высокая загрузка", high_load)


def display_edge_table(extractor, edge_flows, edge_utils):
    """Отображает таблицу загрузки рёбер."""
    st.subheader("🔍 Анализ загрузки рёбер")
    
    data = []
    for i, edge in enumerate(extractor.edges):
        cap = edge.capacity if edge.capacity != float('inf') else float('inf')
        flow = edge_flows[i] if i < len(edge_flows) else 0
        util = edge_utils[i] * 100 if i < len(edge_utils) else 0
        
        if util > 100:
            status = "🔴"
        elif util > 95:
            status = "🟠"
        elif util > 70:
            status = "🟡"
        else:
            status = "🟢"
        
        data.append({
            "": status,
            "Ребро": f"{edge.nodes[0].name} ↔ {edge.nodes[1].name}",
            "Поток (кВт)": f"{flow:,.2f}",
            "Максимум (кВт)": f"{cap:,.2f}" if cap != float('inf') else "∞",
            "Загрузка": f"{util:.1f}%"
        })
    
    df = pd.DataFrame(data)
    df['_sort'] = [float(u.replace('%', '')) for u in df['Загрузка']]
    df = df.sort_values('_sort', ascending=False).drop(columns=['_sort'])
    
    show_all = st.checkbox("Показать все рёбра", value=False)
    if not show_all:
        df = df[df[''].isin(['🔴', '🟠', '🟡'])]
    
    st.dataframe(df, width='stretch', hide_index=True)


def main():
    """Главная функция Streamlit приложения."""
    
    st.markdown('<p class="main-header">⚡ Система оптимизации распределения электрической энергии</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Электрическая сеть «Альфа» • Транснефть</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Режим работы")
        st.markdown("---")
        
        mode = st.radio(
            "Выберите режим:",
            [
                "🧠 ML-обучение",
                "📊 ML-предсказание",
                "🚀 Точное решение (ML + солвер)"
            ]
        )
        
        st.markdown("---")
        st.header("📁 Данные")
        
        use_default = st.checkbox("Использовать стандартные данные", value=True)
        
        if use_default:
            edges_path = "settings/edges.json"
            flows_path = "settings/flows.json"
            st.success("✓ Стандартные данные")
        else:
            edges_file = st.file_uploader("Рёбра графа (JSON)", type=["json"])
            flows_file = st.file_uploader("Заявки (JSON)", type=["json"])
            
            if edges_file and flows_file:
                os.makedirs("uploads", exist_ok=True)
                edges_path = f"uploads/edges_{int(time.time())}.json"
                flows_path = f"uploads/flows_{int(time.time())}.json"
                with open(edges_path, 'wb') as f:
                    f.write(edges_file.getbuffer())
                with open(flows_path, 'wb') as f:
                    f.write(flows_file.getbuffer())
                st.success("✓ Файлы загружены")
            else:
                st.warning("Загрузите оба файла")
                st.stop()
        
        # === РЕЖИМ 1: ML-обучение ===
        if mode == "🧠 ML-обучение":
            st.markdown("---")
            st.header("🎛️ Параметры обучения")
            
            with st.expander("📊 Данные", expanded=False):
                num_samples = st.slider("Сэмплов на уровень", 100, 5000, 500, 100)
                sparsity_options = st.multiselect(
                    "Уровни разреженности",
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    default=[0.1, 0.3, 0.5, 0.7, 0.9]
                )
                demand_scales = st.multiselect(
                    "Масштабы заявок",
                    [0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.5, 1.0],
                    default=[0.02, 0.04]
                )
            
            with st.expander("🏋️ Обучение", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    batch_size = st.selectbox("Размер батча", [32, 64, 128, 256], index=2)
                    epochs = st.slider("Количество эпох", 10, 500, 100, 10)
                    lr = st.selectbox(
                        "Learning rate",
                        [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                        index=3,
                        format_func=lambda x: f"{x:.0e}"
                    )
                with col2:
                    patience = st.slider("Терпение (early stopping)", 3, 50, 10)
                    min_delta = st.selectbox(
                        "Min delta",
                        [1e-6, 1e-5, 1e-4, 1e-3],
                        index=2,
                        format_func=lambda x: f"{x:.0e}"
                    )
            
            with st.expander("📉 Функция потерь", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    demand_weight = st.number_input("demand_weight", 0.1, 1000.0, 10.0, 0.5)
                with col2:
                    excess_weight = st.number_input("excess_weight", 0.1, 100.0, 1.5, 0.5)
                with col3:
                    capacity_weight = st.number_input("capacity_weight", 0.1, 100.0, 1.8, 0.5)
            
            with st.expander("🏗️ Архитектура", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    h1 = st.selectbox("Слой 1", [256, 512, 768, 1024], index=1)
                    h2 = st.selectbox("Слой 2", [128, 256, 512, 768], index=1)
                with col2:
                    h3 = st.selectbox("Слой 3", [64, 128, 256, 512], index=1)
                    dropout = st.slider("Dropout", 0.0, 0.7, 0.3, 0.05)
        
        # === РЕЖИМ 2: ML-предсказание ===
        elif mode == "📊 ML-предсказание":
            st.markdown("---")
            st.header("🤖 Параметры предсказания")
            model_path = st.text_input(
                "Путь к сохранённой модели",
                value="genereted/model.pt"
            )
        
        # === РЕЖИМ 3: Точное решение ===
        elif mode == "🚀 Точное решение (ML + солвер)":
            st.markdown("---")
            st.header("🔧 Параметры солвера")
            
            with st.expander("Градиентный спуск", expanded=True):
                solver_lr = st.selectbox(
                    "Learning rate",
                    [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
                    index=3,
                    format_func=lambda x: f"{x:.2f}"
                )
                solver_max_iter = st.slider("Макс. итераций", 100, 10000, 1000, 100)
                solver_epsilon = st.selectbox(
                    "Эпсилон сходимости",
                    [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                    index=2,
                    format_func=lambda x: f"{x:.0e}"
                )
                solver_patience = st.slider("Терпение", 5, 100, 20)
                solver_verbose = st.checkbox("Подробный вывод", value=True)
            
            st.markdown("---")
            st.header("🤖 ML-начальное приближение")
            solver_model_path = st.text_input(
                "Путь к ML-модели",
                value="genereted/model.pt"
            )
        
        st.markdown("---")
        run_button = st.button("🚀 ЗАПУСТИТЬ", type="primary", width='stretch')
    
    if not run_button:
        st.info("👈 Выберите режим и нажмите «ЗАПУСТИТЬ»")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🧠 ML-обучение")
            st.markdown("Обучить нейросеть с нуля")
        with col2:
            st.markdown("### 📊 ML-предсказание")
            st.markdown("Получить предсказание модели")
        with col3:
            st.markdown("### 🚀 Точное решение")
            st.markdown("ML + градиентный спуск")
        return
    
    # Загружаем граф
    with st.spinner("🔄 Загрузка графа..."):
        graph, registry = load_graph_and_registry(edges_path)
    st.success(f"✓ Граф загружен: {len(graph.nodes)} узлов, {len(graph.edges)} рёбер")
    
    # ============================================================
    # РЕЖИМ 1: ML-обучение
    # ============================================================
    if mode == "🧠 ML-обучение":
        from ml.pipeline import run_training
        
        st.subheader("🧠 Обучение ML-модели")
        
        config = {
            "training": {
                "num_samples_per_level": num_samples,
                "sparsity_levels": sparsity_options,
                "demand_scale_factors": demand_scales,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": lr,
                "early_stopping_patience": patience,
                "min_delta": min_delta
            },
            "model": {"hidden_dims": [h1, h2, h3], "dropout_rate": dropout},
            "loss": {
                "demand_weight": demand_weight,
                "excess_weight": excess_weight,
                "capacity_weight": capacity_weight
            },
            "paths": {"generated_folder": "genereted", "model_save_name": "model.pt"}
        }
        
        with open('settings/config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        train_cfg = ConfigParser('settings/config.json')
        
        run_config = {
            "edges_file": os.path.basename(edges_path),
            "flows_file": os.path.basename(flows_path),
            "mode": "train",
            "visualize_flows": False
        }
        with open('settings/run_config.json', 'w', encoding='utf-8') as f:
            json.dump(run_config, f, indent=2)
        run_cfg = ConfigParser('settings/run_config.json')
        
        os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)
        
        status_container = st.empty()
        progress_bar = st.progress(0)
        log_placeholder = st.empty()
        epoch_placeholder = st.empty()
        
        log_buffer = io.StringIO()
        
        class TeeOutput:
            def __init__(self, buffer):
                self.buffer = buffer
                self.terminal = sys.stdout
            def write(self, message):
                self.terminal.write(message)
                self.terminal.flush()
                self.buffer.write(message)
            def flush(self):
                self.terminal.flush()
        
        tee = TeeOutput(log_buffer)
        original_stdout = sys.stdout
        sys.stdout = tee
        
        try:
            status_container.text("⏳ Генерация данных и обучение...")
            progress_bar.progress(5)
            
            training_done = threading.Event()
            
            def train_thread():
                run_training(graph, registry, run_cfg, train_cfg)
                training_done.set()
            
            thread = threading.Thread(target=train_thread)
            thread.start()
            
            epoch_info = {"current": 0, "total": epochs}
            
            while not training_done.is_set():
                thread.join(timeout=0.5)
                current_logs = log_buffer.getvalue()
                log_lines = current_logs.split('\n')
                epoch_lines = [l for l in log_lines if 'Epoch' in l and '|' in l]
                
                if epoch_lines:
                    last_epoch = epoch_lines[-1]
                    try:
                        parts = last_epoch.split('|')
                        epoch_num = int(parts[0].replace('Epoch', '').strip())
                        epoch_info["current"] = epoch_num
                    except:
                        pass
                
                pct = min(epoch_info["current"] / max(epochs, 1), 1.0)
                progress_bar.progress(int(5 + pct * 85))
                
                if epoch_info["current"] > 0:
                    epoch_placeholder.markdown(
                        f"""
                        <div class="epoch-badge">
                            <b>Эпоха {epoch_info["current"]} / {epochs}</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                other_lines = [l for l in log_lines[-30:] if 'Epoch' not in l or '|' not in l]
                if other_lines:
                    newline = "\n"
                    log_placeholder.markdown(
                        f'<div class="training-log"><pre>{newline.join(other_lines[-10:])}</pre></div>',
                        unsafe_allow_html=True
                    )
            
            final_logs = log_buffer.getvalue()
            progress_bar.progress(95)
            status_container.text("✅ Обучение завершено!")
            epoch_placeholder.empty()
            
            with st.expander("📝 Полные логи обучения"):
                st.text_area("Логи", final_logs, height=300)
            
            # Графики
            gen_folder = train_cfg.paths.generated_folder
            
            loss_file = f"{gen_folder}/loss_curves.png"
            comp_file = f"{gen_folder}/loss_components.png"
            
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(loss_file):
                    st.image(Image.open(loss_file), caption="Кривые обучения", width='stretch')
            with col2:
                if os.path.exists(comp_file):
                    st.image(Image.open(comp_file), caption="Компоненты функции потерь", width='stretch')
            
            progress_bar.progress(100)
            st.success(f"✅ Модель сохранена в `{gen_folder}/{train_cfg.paths.model_save_name}`")
        
        finally:
            sys.stdout = original_stdout
    
    # ============================================================
    # РЕЖИМ 2: ML-предсказание
    # ============================================================
    elif mode == "📊 ML-предсказание":
        from ml.pipeline import run_prediction
        
        st.subheader("📊 ML-предсказание")
        
        # Загружаем конфиг (минимальный)
        config = {
            "model": {"hidden_dims": [512, 256, 128], "dropout_rate": 0.3},
            "paths": {"generated_folder": "genereted", "model_save_name": "model.pt"}
        }
        with open('settings/config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        train_cfg = ConfigParser('settings/config.json')
        
        run_config = {
            "edges_file": os.path.basename(edges_path),
            "flows_file": os.path.basename(flows_path),
            "mode": "predict",
            "model_path": model_path,
            "visualize_flows": True
        }
        with open('settings/run_config.json', 'w', encoding='utf-8') as f:
            json.dump(run_config, f, indent=2)
        run_cfg = ConfigParser('settings/run_config.json')
        
        os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)
        
        with st.spinner("🔄 Загрузка модели и предсказание..."):
            import torch
            from ml.feature_extractor import FeatureExtractor
            from ml.model import PathWeightNetwork
            from ml.loss import EdgeFlowCalculator
            from ml.inference import FlowPredictor
            
            extractor = FeatureExtractor(graph, registry)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model = PathWeightNetwork(
                input_dim=checkpoint['feature_dim'],
                output_shape=checkpoint['output_shape'],
                hidden_dims=tuple(checkpoint.get('hidden_dims', [512, 256, 128])),
                dropout_rate=checkpoint.get('dropout_rate', 0.3)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.set_path_mask(checkpoint['path_mask'])
            model.to(device).eval()
            
            edge_calc = EdgeFlowCalculator(registry, extractor)
            predictor = FlowPredictor(model, extractor, edge_calc, device)
            
            with open(f"settings/{run_cfg.flows_file}", 'r', encoding='utf-8') as f:
                base_flows = json.load(f)
            
            raw_real = extractor.build_raw_features(base_flows)
            real_feat, real_mask = extractor.normalize_features(raw_real)
            results = predictor.predict_with_normalized(real_feat, base_flows, real_mask)
        
        # Метрики
        st.markdown("---")
        st.subheader("📊 Результаты ML-предсказания")
        
        edge_utils = results.get('edge_utilization', np.array([]))
        edge_flows = results.get('edge_flows', np.array([]))
        
        display_metrics(
            results.get('demanded', 0),
            results.get('total_delivered', 0),
            edge_utils
        )
        
        st.markdown("---")
        display_edge_table(extractor, edge_flows, edge_utils)
        
        # Граф
        st.markdown("---")
        st.subheader("🔗 Граф потоков")
        
        ml_graph_path = f"{train_cfg.paths.generated_folder}/ml_prediction.html"
        if os.path.exists(ml_graph_path):
            with open(ml_graph_path, 'r', encoding='utf-8') as f:
                st.components.v1.html(f.read(), height=700, scrolling=True)
        else:
            # Запускаем run_prediction для создания графа
            run_prediction(graph, registry, run_cfg, train_cfg)
            if os.path.exists(ml_graph_path):
                with open(ml_graph_path, 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=700, scrolling=True)
            else:
                st.warning("Файл графа не найден")
    
    # ============================================================
    # РЕЖИМ 3: Точное решение (ML + солвер)
    # ============================================================
    elif mode == "🚀 Точное решение (ML + солвер)":
        from ml.pipeline import run_solver_pipeline
        
        st.subheader("🚀 Точное решение (ML + солвер)")
        
        config = {
            "model": {"hidden_dims": [512, 256, 128], "dropout_rate": 0.3},
            "solver": {
                "learning_rate": solver_lr,
                "max_iter": solver_max_iter,
                "epsilon": solver_epsilon,
                "early_stopping_patience": solver_patience,
                "gradient_epsilon_rel": 0.01,
                "capacity_weight": 1.8,
                "verbose": solver_verbose
            },
            "paths": {"generated_folder": "genereted", "model_save_name": "model.pt"}
        }
        
        with open('settings/config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        train_cfg = ConfigParser('settings/config.json')
        
        run_config = {
            "edges_file": os.path.basename(edges_path),
            "flows_file": os.path.basename(flows_path),
            "mode": "solve",
            "use_ml_initial_guess": True,
            "model_path": solver_model_path,
            "visualize_flows": True
        }
        with open('settings/run_config.json', 'w', encoding='utf-8') as f:
            json.dump(run_config, f, indent=2)
        run_cfg = ConfigParser('settings/run_config.json')
        
        os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)
        
        status_container = st.empty()
        progress_bar = st.progress(0)
        log_placeholder = st.empty()
        iter_placeholder = st.empty()
        
        log_buffer = io.StringIO()
        
        class TeeOutput:
            def __init__(self, buffer):
                self.buffer = buffer
                self.terminal = sys.stdout
            def write(self, message):
                self.terminal.write(message)
                self.terminal.flush()
                self.buffer.write(message)
            def flush(self):
                self.terminal.flush()
        
        tee = TeeOutput(log_buffer)
        original_stdout = sys.stdout
        sys.stdout = tee
        
        try:
            status_container.text("⏳ Расчёт...")
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
            
            iter_info = {"current": 0, "loss": 0.0}
            
            while not solver_done.is_set():
                thread.join(timeout=0.3)
                current_logs = log_buffer.getvalue()
                log_lines = current_logs.split('\n')
                iter_lines = [l for l in log_lines if 'Итерация' in l and 'loss' in l.lower()]
                
                if iter_lines:
                    last_iter = iter_lines[-1]
                    try:
                        parts = last_iter.split(':')
                        iter_num = int(parts[0].replace('Итерация', '').strip())
                        loss_part = parts[1].split('=')[1].split('кВт')[0].strip()
                        iter_info["current"] = iter_num
                        iter_info["loss"] = float(loss_part)
                    except:
                        pass
                
                pct = min(iter_info["current"] / max(solver_max_iter, 1), 1.0)
                progress_bar.progress(int(10 + pct * 80))
                
                if iter_info["current"] > 0:
                    iter_placeholder.markdown(
                        f"""
                        <div class="solver-badge">
                            <b>Итерация {iter_info["current"]} / {solver_max_iter}</b><br>
                            <small>Текущий loss: <b>{iter_info["loss"]:.2f} кВт</b></small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                other_lines = [l for l in log_lines[-30:] if 'Итерация' not in l]
                if other_lines:
                    newline = "\n"
                    log_placeholder.markdown(
                        f'<div class="training-log"><pre>{newline.join(other_lines[-10:])}</pre></div>',
                        unsafe_allow_html=True
                    )
            
            final_logs = log_buffer.getvalue()
            progress_bar.progress(95)
            status_container.text("✅ Расчёт завершён!")
            iter_placeholder.empty()
            
            result = solver_result.get('result')
            solver = solver_result.get('solver')
            
            with st.expander("📝 Полные логи расчёта"):
                st.text_area("Логи", final_logs, height=300)
            
            if result and solver:
                delivery = solver.get_delivery_report()
                
                st.markdown("---")
                st.subheader("📊 Результаты точного решения")
                
                edge_loads = solver.get_edge_loads()
                edge_utils_arr = np.array([r for _, r in edge_loads.values()])
                
                display_metrics(
                    delivery['total_requested'],
                    delivery['total_delivered'],
                    edge_utils_arr
                )
                
                st.markdown("---")
                # Таблица рёбер
                st.subheader("🔍 Анализ загрузки рёбер")
                
                edge_data = []
                for edge, (flow, ratio) in edge_loads.items():
                    cap = edge.capacity if edge.capacity != float('inf') else float('inf')
                    util = ratio * 100
                    
                    if util > 100:
                        status = "🔴"
                    elif util > 95:
                        status = "🟠"
                    elif util > 70:
                        status = "🟡"
                    else:
                        status = "🟢"
                    
                    edge_data.append({
                        "": status,
                        "Ребро": f"{edge.nodes[0].name} ↔ {edge.nodes[1].name}",
                        "Поток (кВт)": f"{flow:,.2f}",
                        "Максимум (кВт)": f"{cap:,.2f}" if cap != float('inf') else "∞",
                        "Загрузка": f"{util:.1f}%"
                    })
                
                df_edges = pd.DataFrame(edge_data)
                df_edges['_sort'] = [float(u.replace('%', '')) for u in df_edges['Загрузка']]
                df_edges = df_edges.sort_values('_sort', ascending=False).drop(columns=['_sort'])
                
                show_all = st.checkbox("Показать все рёбра", value=False)
                if not show_all:
                    df_edges = df_edges[df_edges[''].isin(['🔴', '🟠', '🟡'])]
                
                st.dataframe(df_edges, width='stretch', hide_index=True)
                
                # Граф
                st.markdown("---")
                st.subheader("🔗 Граф потоков")
                
                output_path = f"{train_cfg.paths.generated_folder}/solution_graph.html"
                if os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as f:
                        st.components.v1.html(f.read(), height=700, scrolling=True)
                else:
                    st.warning("Файл графа не найден")
                
                # График обучения солвера
                st.markdown("---")
                st.subheader("📈 Обучение солвера")
                history_file = f"{train_cfg.paths.generated_folder}/solver_history.png"
                if os.path.exists(history_file):
                    st.image(Image.open(history_file), width='stretch')
            
            progress_bar.progress(100)
        
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()