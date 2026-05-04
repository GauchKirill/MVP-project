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

# Добавляем src в путь для импортов
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config_parser import ConfigParser
from graph import Graph, GraphView, RequestRegistry
from solver import Solver, FlowsCreator
from ml.feature_extractor import FeatureExtractor

# Настройки страницы
st.set_page_config(
    page_title="Транснефть: Оптимизация потоков",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
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


def display_delivery_report(delivery: dict):
    """Отображает отчёт о доставке энергии."""
    st.subheader("📦 Доставка энергии по заявкам")
    
    if not delivery.get('items'):
        st.info("Нет данных о доставке")
        return
    
    data = []
    for item in delivery['items']:
        shortage_pct = item.get('shortage_pct', 0)
        status = "✓" if shortage_pct < 1 else "⚠️" if shortage_pct < 10 else "❌"
        
        data.append({
            "": status,
            "Источник": item['source'],
            "Потребитель": item['consumer'],
            "Заявлено (кВт)": f"{item['requested']:,.2f}",
            "Доставлено (кВт)": f"{item['delivered']:,.2f}",
            "Недопоставка": f"{item['shortage']:,.2f}",
            "%": f"{shortage_pct:.1f}"
        })
    
    df = pd.DataFrame(data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего заявок", len(data))
    with col2:
        fully_delivered = sum(1 for d in delivery['items'] if d.get('shortage_pct', 0) < 1)
        st.metric("✓ Доставлено полностью", fully_delivered)
    with col3:
        problems = sum(1 for d in delivery['items'] if d.get('shortage_pct', 0) >= 10)
        st.metric("⚠️ Проблемные", problems)
    
    st.dataframe(df, width='stretch', hide_index=True)


def main():
    """Главная функция Streamlit приложения."""
    
    st.markdown('<p class="main-header">⚡ Система оптимизации распределения электрической энергии</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Электрическая сеть «Альфа» • Транснефть</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Режим работы")
        st.markdown("---")
        
        mode = st.radio(
            "Выберите режим:",
            [
                "🧠 ML-приближение (обучить модель)",
                "🎯 Точный расчёт (солвер)",
                "🚀 Полный пайплайн (ML + солвер)"
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
        
        # Настройки ML
        if "ML" in mode:
            st.markdown("---")
            st.header("🎛️ Параметры обучения")
            
            with st.expander("📊 Данные", expanded=False):
                num_samples = st.slider("Сэмплов на уровень", 100, 5000, 500, 100)
                sparsity_options = st.multiselect(
                    "Уровни разреженности",
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    default=[0.1, 0.3, 0.5, 0.7, 0.9]
                )
                demand_scale_factors = st.multiselect(
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
                        "Min delta (чувствительность)",
                        [1e-6, 1e-5, 1e-4, 1e-3],
                        index=2,
                        format_func=lambda x: f"{x:.0e}"
                    )
                    grad_eps = st.selectbox(
                        "Gradient epsilon (солвер)",
                        [0.001, 0.01, 0.05, 0.1],
                        index=1,
                        format_func=lambda x: f"{x:.3f}"
                    )
            
            with st.expander("📉 Функция потерь", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    demand_weight = st.number_input("demand_weight", 0.1, 1000.0, 10.0, 0.5)
                with col2:
                    excess_weight = st.number_input("excess_weight", 0.1, 100.0, 1.5, 0.5)
                with col3:
                    capacity_weight = st.number_input("capacity_weight", 0.1, 100.0, 1.8, 0.5)
            
            with st.expander("🏗️ Архитектура модели", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    hidden_dim1 = st.selectbox("Скрытый слой 1", [256, 512, 768, 1024], index=1)
                    hidden_dim2 = st.selectbox("Скрытый слой 2", [128, 256, 512, 768], index=1)
                with col2:
                    hidden_dim3 = st.selectbox("Скрытый слой 3", [64, 128, 256, 512], index=1)
                    dropout_rate = st.slider("Dropout", 0.0, 0.7, 0.3, 0.05)
        
        # Настройки солвера
        if "солвер" in mode or "пайплайн" in mode:
            st.markdown("---")
            st.header("🔧 Параметры солвера")
            
            with st.expander("Градиентный спуск", expanded=False):
                solver_lr = st.selectbox(
                    "Learning rate (солвер)",
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
                solver_patience = st.slider("Терпение (солвер)", 5, 100, 20)
                solver_verbose = st.checkbox("Подробный вывод", value=True)
        else:
            solver_lr = 0.3
            solver_max_iter = 1000
            solver_epsilon = 1e-4
            solver_patience = 20
            solver_verbose = True
        
        st.markdown("---")
        run_button = st.button("🚀 ЗАПУСТИТЬ", type="primary", width='stretch')
    
    # Основная область
    if not run_button:
        st.info("👈 Выберите режим и нажмите «ЗАПУСТИТЬ»")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🧠 ML-приближение")
            st.markdown("Обучить нейросеть для быстрых предсказаний")
        with col2:
            st.markdown("### 🎯 Точный расчёт")
            st.markdown("Градиентный спуск для точного решения")
        with col3:
            st.markdown("### 🚀 Полный пайплайн")
            st.markdown("ML-приближение + уточнение солвером")
        return
    
    # Запуск
    with st.spinner("🔄 Загрузка графа..."):
        graph, registry = load_graph_and_registry(edges_path)
    
    st.success(f"✓ Граф загружен: {len(graph.nodes)} узлов, {len(graph.edges)} рёбер")
    
    # Формируем конфиг
    config = {
        "training": {
            "num_samples_per_level": num_samples if 'num_samples' in dir() else 500,
            "sparsity_levels": sparsity_options if 'sparsity_options' in dir() else [0.1, 0.3, 0.5, 0.7, 0.9],
            "demand_scale_factors": demand_scale_factors if 'demand_scale_factors' in dir() else [0.02, 0.04],
            "batch_size": batch_size if 'batch_size' in dir() else 128,
            "epochs": epochs if 'epochs' in dir() else 100,
            "learning_rate": lr if 'lr' in dir() else 1e-3,
            "early_stopping_patience": patience if 'patience' in dir() else 10,
            "min_delta": min_delta if 'min_delta' in dir() else 1e-4,
            "gradient_epsilon_rel": grad_eps if 'grad_eps' in dir() else 0.01
        },
        "model": {
            "hidden_dims": [
                hidden_dim1 if 'hidden_dim1' in dir() else 512,
                hidden_dim2 if 'hidden_dim2' in dir() else 256,
                hidden_dim3 if 'hidden_dim3' in dir() else 128
            ],
            "dropout_rate": dropout_rate if 'dropout_rate' in dir() else 0.3
        },
        "loss": {
            "demand_weight": demand_weight if 'demand_weight' in dir() else 10.0,
            "excess_weight": excess_weight if 'excess_weight' in dir() else 1.5,
            "capacity_weight": capacity_weight if 'capacity_weight' in dir() else 1.8
        },
        "solver": {
            "learning_rate": solver_lr,
            "max_iter": solver_max_iter,
            "epsilon": solver_epsilon,
            "early_stopping_patience": solver_patience,
            "gradient_epsilon_rel": grad_eps if 'grad_eps' in dir() else 0.01,
            "capacity_weight": capacity_weight if 'capacity_weight' in dir() else 1.8,
            "verbose": solver_verbose
        },
        "visualization": {
            "training": True,
            "flows": True,
            "save_report": True,
            "visualize_data": True
        },
        "paths": {
            "generated_folder": "genereted",
            "model_save_name": "model.pt",
            "graph_html": "graph.html"
        }
    }
    
    with open('settings/config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    train_cfg = ConfigParser('settings/config.json')
    
    run_config = {
        "edges_file": os.path.basename(edges_path),
        "flows_file": os.path.basename(flows_path),
        "visualize_flows": True,
    }
    
    if "ML-приближение" in mode:
        run_config["mode"] = "train"
    elif "Точный расчёт" in mode:
        run_config["mode"] = "solve"
        run_config["use_ml_initial_guess"] = False
        run_config["model_path"] = f"{train_cfg.paths.generated_folder}/{train_cfg.paths.model_save_name}"
    elif "Полный пайплайн" in mode:
        run_config["mode"] = "solve"
        run_config["use_ml_initial_guess"] = True
        run_config["model_path"] = f"{train_cfg.paths.generated_folder}/{train_cfg.paths.model_save_name}"
    
    with open('settings/run_config.json', 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2)
    
    run_cfg = ConfigParser('settings/run_config.json')
    os.makedirs(train_cfg.paths.generated_folder, exist_ok=True)
    
    # Выполнение
    try:
        if run_cfg.mode == "train":
            from ml.pipeline import run_training
            
            st.subheader("🧠 Обучение ML-модели")
            
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
                
                epoch_info = {"current": 0, "total": train_cfg.training.epochs, "status": ""}
                
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
                            epoch_info["status"] = last_epoch.strip()
                        except:
                            pass
                    
                    pct = min(epoch_info["current"] / max(epoch_info["total"], 1), 1.0)
                    progress_bar.progress(int(5 + pct * 85))
                    
                    if epoch_info["status"]:
                        epoch_placeholder.markdown(
                            f"""
                            <div class="epoch-badge">
                                <b>Эпоха {epoch_info["current"]} / {epoch_info["total"]}</b><br>
                                <small>{epoch_info["status"]}</small>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    other_lines = [l for l in log_lines[-40:] if 'Epoch' not in l or '|' not in l]
                    if other_lines:
                        newline = "\n"
                        log_placeholder.markdown(
                            f'<div class="training-log"><pre>{newline.join(other_lines[-15:])}</pre></div>',
                            unsafe_allow_html=True
                        )
                
                final_logs = log_buffer.getvalue()
                progress_bar.progress(95)
                status_container.text("✅ Обучение завершено!")
                epoch_placeholder.empty()
                
                with st.expander("📝 Полные логи обучения"):
                    st.text_area("Логи", final_logs, height=300)
                
                # Графики
                st.markdown("---")
                st.subheader("📈 Графики обучения")
                
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
                
                # Тестирование на реальных данных
                st.markdown("---")
                st.subheader("🧪 Тестирование на реальных данных")
                
                with st.spinner("Загрузка модели и тестирование..."):
                    from ml.inference import FlowPredictor
                    from ml.loss import EdgeFlowCalculator
                    from ml.model import PathWeightNetwork
                    import torch
                    
                    extractor = FeatureExtractor(graph, registry)
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    
                    model_path = f"{gen_folder}/{train_cfg.paths.model_save_name}"
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    
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
                    
                    with open(f"settings/{run_cfg.flows_file}", 'r', encoding='utf-8') as f:
                        base_flows = json.load(f)
                    
                    raw_real = extractor.build_raw_features(base_flows)
                    real_feat, real_mask = extractor.normalize_features(raw_real)
                    results = predictor.predict_with_normalized(real_feat, base_flows, real_mask)
                    
                    # Метрики
                    st.markdown("### 📊 Результаты ML-предсказания")
                    
                    edge_utils = results.get('edge_utilization', np.array([]))
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Заявлено", f"{results.get('demanded', 0):,.1f} кВт")
                    with col2:
                        delivered = results.get('total_delivered', 0)
                        demanded = results.get('demanded', 1)
                        ratio = delivered / demanded * 100 if demanded > 0 else 0
                        st.metric("Доставлено (ML)", f"{delivered:,.1f} кВт", delta=f"{ratio:.1f}%")
                    with col3:
                        overloaded = int((edge_utils > 0.95).sum())
                        st.metric("Перегружено рёбер", overloaded, delta="⚠️" if overloaded > 0 else "✓")
                    with col4:
                        high_load = int(((edge_utils > 0.7) & (edge_utils <= 0.95)).sum())
                        st.metric("Высокая загрузка", high_load)
                    
                    # Анализ рёбер
                    st.markdown("---")
                    st.subheader("🔍 Анализ загрузки рёбер")
                    
                    edge_data = []
                    edge_flows_arr = results.get('edge_flows', np.array([]))
                    for i, edge in enumerate(extractor.edges):
                        cap = edge.capacity if edge.capacity != float('inf') else float('inf')
                        flow = edge_flows_arr[i] if i < len(edge_flows_arr) else 0
                        util = edge_utils[i] * 100 if i < len(edge_utils) else 0
                        
                        if util > 95:
                            status = "🔴"
                        elif util > 70:
                            status = "🟠"
                        elif util > 30:
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
                    
                    # show_all = st.checkbox("Показать все рёбра", value=False)
                    # if not show_all:
                    #     df_edges = df_edges[df_edges[''].isin(['🔴', '🟠'])]
                    
                    # st.dataframe(df_edges, width='stretch', hide_index=True)
                    
                    # Граф потоков
                    st.markdown("---")
                    st.subheader("🔗 Граф потоков (ML-предсказание)")
                    
                    ml_graph_path = f"{gen_folder}/ml_prediction.html"
                    if os.path.exists(ml_graph_path):
                        with open(ml_graph_path, 'r', encoding='utf-8') as f:
                            st.components.v1.html(f.read(), height=700, scrolling=True)
                    else:
                        st.warning(f"Файл графа не найден: {ml_graph_path}")
            
            finally:
                sys.stdout = original_stdout
        
        elif run_cfg.mode == "solve":
            from ml.pipeline import run_solver_pipeline
            
            if run_cfg.use_ml_initial_guess:
                st.subheader("🚀 Полный пайплайн (ML + солвер)")
            else:
                st.subheader("🎯 Точный расчёт (солвер)")
            
            status_container = st.empty()
            progress_bar = st.progress(0)
            log_placeholder = st.empty()
            iter_placeholder = st.empty()  # для красивого вывода итераций
            
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
                max_iter = train_cfg.solver.max_iter
                
                solver_done = threading.Event()
                solver_result = {}
                
                def solver_thread():
                    result, s = run_solver_pipeline(graph, registry, run_cfg, train_cfg)
                    solver_result['result'] = result
                    solver_result['solver'] = s
                    solver_done.set()
                
                thread = threading.Thread(target=solver_thread)
                thread.start()
                
                iter_info = {"current": 0, "loss": 0.0}
                
                while not solver_done.is_set():
                    thread.join(timeout=0.3)
                    current_logs = log_buffer.getvalue()
                    log_lines = current_logs.split('\n')
                    
                    # Ищем строки с итерациями
                    iter_lines = [l for l in log_lines if 'Итерация' in l and 'loss' in l.lower()]
                    
                    if iter_lines:
                        last_iter = iter_lines[-1]
                        try:
                            # Парсим: "Итерация  100: loss = 123.45 кВт ..."
                            parts = last_iter.split(':')
                            iter_num = int(parts[0].replace('Итерация', '').strip())
                            loss_part = parts[1].split('=')[1].split('кВт')[0].strip()
                            iter_info["current"] = iter_num
                            iter_info["loss"] = float(loss_part)
                        except:
                            pass
                    
                    pct = min(iter_info["current"] / max(max_iter, 1), 1.0)
                    progress_bar.progress(int(10 + pct * 80))
                    
                    if iter_info["current"] > 0:
                        loss_color = "#27ae60" if iter_info["loss"] < 100 else "#f39c12" if iter_info["loss"] < 1000 else "#e74c3c"
                        iter_placeholder.markdown(
                            f"""
                            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                        color: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                <b>Итерация {iter_info["current"]} / {max_iter}</b><br>
                                <small>Текущий loss: <b style="color: {loss_color};">{iter_info["loss"]:.2f} кВт</b></small>
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
                
                # Финальное состояние
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
                    st.subheader("📊 Ключевые метрики")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Заявлено", f"{delivery['total_requested']:,.1f} кВт")
                    with col2:
                        ratio = delivery['total_delivered'] / delivery['total_requested'] * 100
                        st.metric("Доставлено", f"{delivery['total_delivered']:,.1f} кВт", 
                                 delta=f"{ratio:.1f}%")
                    with col3:
                        violations = solver.get_edge_violations()
                        st.metric("Рёбер с превышением", len(violations),
                                 delta="✓" if not violations else f"⚠️ {len(violations)}")
                    
                    st.markdown("---")
                    tab1, tab2, tab3 = st.tabs([
                        "🔗 Граф потоков", "📋 Доставка", "📈 Обучение солвера"
                    ])
                    
                    with tab1:
                        output_path = f"{train_cfg.paths.generated_folder}/solution_graph.html"
                        if os.path.exists(output_path):
                            with open(output_path, 'r', encoding='utf-8') as f:
                                st.components.v1.html(f.read(), height=700, scrolling=True)
                        else:
                            st.warning("Файл графа не найден.")
                    
                    with tab2:
                        display_delivery_report(delivery)
                    
                    with tab3:
                        history_file = f"{train_cfg.paths.generated_folder}/solver_history.png"
                        if os.path.exists(history_file):
                            st.image(Image.open(history_file), width='stretch')
                        else:
                            st.warning("График обучения солвера не найден")
                
                progress_bar.progress(100)
                status_container.text("✅ Расчёт завершён!")
            
            finally:
                sys.stdout = original_stdout
    
    except Exception as e:
        st.error(f"❌ Ошибка: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
