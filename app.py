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
from pathlib import Path
from contextlib import redirect_stdout
from PIL import Image

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
            "Статус": status,
            "Источник": item['source'],
            "Потребитель": item['consumer'],
            "Заявлено (кВт)": f"{item['requested']:,.2f}",
            "Доставлено (кВт)": f"{item['delivered']:,.2f}",
            "Недопоставка (кВт)": f"{item['shortage']:,.2f}",
            "Недопоставка (%)": f"{shortage_pct:.1f}%"
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
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Скачать отчёт (CSV)", csv, "delivery_report.csv", "text/csv")


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
        
        # Настройки обучения
        if "ML" in mode:
            st.markdown("---")
            st.header("🎛️ Параметры обучения")
            
            with st.expander("Данные", expanded=False):
                num_samples = st.slider("Сэмплов на уровень", 100, 5000, 500, 100)
                sparsity_options = st.multiselect(
                    "Уровни разреженности",
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    default=[0.1, 0.3, 0.5, 0.7, 0.9]
                )
                demand_scales = st.multiselect(
                    "Масштабы заявок",
                    [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                    default=[0.5, 1.0]
                )
            
            with st.expander("Обучение", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    batch_size = st.selectbox("Размер батча", [32, 64, 128, 256], index=1)
                    epochs = st.slider("Количество эпох", 10, 500, 100, 10)
                with col2:
                    lr = st.selectbox(
                        "Learning rate",
                        [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                        index=2,
                        format_func=lambda x: f"{x:.0e}"
                    )
                    patience = st.slider("Терпение (early stopping)", 3, 50, 10)
            
            with st.expander("Функция потерь", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    demand_weight = st.number_input("demand_weight", 0.1, 1000.0, 10.0, 0.5)
                with col2:
                    excess_weight = st.number_input("excess_weight", 0.1, 100.0, 1.5, 0.5)
                with col3:
                    capacity_weight = st.number_input("capacity_weight", 0.1, 100.0, 1.8, 0.5)
        
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
        else:
            solver_lr = 0.3
            solver_max_iter = 1000
            solver_epsilon = 1e-4
        
        st.markdown("---")
        run_button = st.button("🚀 ЗАПУСТИТЬ", type="primary", use_container_width=True)
    
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
    if "ML" in mode:
        config = {
            "training": {
                "num_samples_per_level": num_samples,
                "sparsity_levels": sparsity_options,
                "demand_scale_factors": demand_scales,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": lr,
                "early_stopping_patience": patience,
                "min_delta": 1e-4
            },
            "model": {
                "hidden_dims": [512, 256, 128],
                "dropout_rate": 0.3
            },
            "loss": {
                "demand_weight": demand_weight,
                "excess_weight": excess_weight,
                "capacity_weight": capacity_weight
            },
            "solver": {
                "learning_rate": solver_lr,
                "max_iter": solver_max_iter,
                "epsilon": solver_epsilon,
                "gradient_epsilon_rel": 0.01,
                "verbose": False
            },
            "visualization": {
                "training": True,
                "flows": True,
                "save_report": True,
                "visualize_data": False
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
            from ml.inference import FlowPredictor
            from ml.loss import EdgeFlowCalculator
            import torch
            
            st.subheader("🧠 Обучение ML-модели")
            
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            status_container.text("⏳ Генерация данных и обучение...")
            progress_bar.progress(10)
            
            log_stream = io.StringIO()
            
            with redirect_stdout(log_stream):
                result = run_training(graph, registry, run_cfg, train_cfg)
            
            progress_bar.progress(90)
            status_container.text("✅ Обучение завершено!")
            
            # Логи
            st.markdown("---")
            st.subheader("📝 Логи обучения")
            st.text_area("Вывод", log_stream.getvalue(), height=300)
            
            # Графики
            st.markdown("---")
            st.subheader("📈 Графики обучения")
            
            gen_folder = train_cfg.paths.generated_folder
            
            history_file = f"{gen_folder}/training_history.png"
            if os.path.exists(history_file):
                st.image(Image.open(history_file), caption="История обучения", use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                pca_file = f"{gen_folder}/pca_training_data.png"
                if os.path.exists(pca_file):
                    st.image(Image.open(pca_file), caption="PCA", use_container_width=True)
            with col2:
                dist_file = f"{gen_folder}/distribution_training_data.png"
                if os.path.exists(dist_file):
                    st.image(Image.open(dist_file), caption="Распределение", use_container_width=True)
            
            progress_bar.progress(100)
            st.success(f"✅ Модель сохранена в `{gen_folder}/{train_cfg.paths.model_save_name}`")
            
            # Тестирование
            st.markdown("---")
            st.subheader("🧪 Тестирование на реальных данных")
            
            with st.spinner("Загрузка модели..."):
                extractor = FeatureExtractor(graph, registry)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                model_path = f"{gen_folder}/{train_cfg.paths.model_save_name}"
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                
                from ml.model import PathWeightNetwork
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
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Заявлено", f"{results.get('demanded', 0):,.1f} кВт")
                with col2:
                    delivered = results.get('total_delivered', 0)
                    demanded = results.get('demanded', 1)
                    ratio = delivered / demanded * 100 if demanded > 0 else 0
                    st.metric("Доставлено (ML)", f"{delivered:,.1f} кВт", delta=f"{ratio:.1f}%")
                with col3:
                    edge_utils = results.get('edge_utilization', np.array([]))
                    overloaded = (edge_utils > 0.95).sum()
                    st.metric("Перегружено рёбер", int(overloaded))
        
        elif run_cfg.mode == "solve":
            from ml.pipeline import run_solver_pipeline
            
            if run_cfg.use_ml_initial_guess:
                st.subheader("🚀 Полный пайплайн (ML + солвер)")
            else:
                st.subheader("🎯 Точный расчёт (солвер)")
            
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            status_container.text("⏳ Расчёт...")
            progress_bar.progress(30)
            
            log_stream = io.StringIO()
            
            with redirect_stdout(log_stream):
                result, solver = run_solver_pipeline(graph, registry, run_cfg, train_cfg)
            
            progress_bar.progress(70)
            
            if result:
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
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🔗 Граф потоков", "📋 Доставка", "📈 Обучение", "📝 Логи"
                ])
                
                with tab1:
                    output_path = f"{train_cfg.paths.generated_folder}/solution_graph.html"
                    if os.path.exists(output_path):
                        with open(output_path, 'r', encoding='utf-8') as f:
                            st.components.v1.html(f.read(), height=700, scrolling=True)
                
                with tab2:
                    display_delivery_report(delivery)
                
                with tab3:
                    history_file = f"{train_cfg.paths.generated_folder}/solver_history.png"
                    if os.path.exists(history_file):
                        st.image(Image.open(history_file), use_container_width=True)
                
                with tab4:
                    st.text_area("Логи", log_stream.getvalue(), height=300)
            
            progress_bar.progress(100)
            status_container.text("✅ Расчёт завершён!")
    
    except Exception as e:
        st.error(f"❌ Ошибка: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
