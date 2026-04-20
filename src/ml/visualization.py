"""Визуализация процесса обучения и результатов."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import json
from datetime import datetime


class TrainingVisualizer:
    """
    Визуализация процесса обучения нейросети.
    """
    
    def __init__(self, save_dir: str = 'genereted'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_history(self, 
                            history: Dict,
                            save_name: str = 'training_history.png',
                            show: bool = True) -> None:
        """
        Строит графики обучения: loss и компоненты.
        
        Args:
            history: словарь с историей обучения из ModelTrainer
            save_name: имя файла для сохранения
            show: показать ли график
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(len(history['train_loss']))
        
        # 1. Общий лосс
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Loss')
        ax.set_title('Общая функция потерь')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Отмечаем минимум валидационного лосса
        if history['val_loss']:
            min_val_epoch = np.argmin(history['val_loss'])
            min_val_loss = history['val_loss'][min_val_epoch]
            ax.plot(min_val_epoch, min_val_loss, 'r*', markersize=10, 
                   label=f'Мин. val: {min_val_loss:.4f}')
            ax.legend()
        
        # 2. Компоненты лосса на train
        ax = axes[0, 1]
        if history['train_components']:
            # Собираем данные по компонентам
            components = {}
            for comp_dict in history['train_components']:
                for key, value in comp_dict.items():
                    if key not in components:
                        components[key] = []
                    components[key].append(value)
            
            for key, values in components.items():
                ax.plot(epochs, values, label=key, linewidth=1.5)
            
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Значение компонента')
            ax.set_title('Компоненты лосса (Train)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Компоненты лосса на validation
        ax = axes[1, 0]
        if history['val_components']:
            components = {}
            for comp_dict in history['val_components']:
                for key, value in comp_dict.items():
                    if key not in components:
                        components[key] = []
                    components[key].append(value)
            
            for key, values in components.items():
                ax.plot(epochs, values, label=key, linewidth=1.5)
            
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Значение компонента')
            ax.set_title('Компоненты лосса (Validation)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Дополнительные метрики
        ax = axes[1, 1]
        if history['val_components']:
            # Достаём delivery_ratio и avg_utilization
            delivery_ratios = []
            avg_utils = []
            for comp_dict in history['val_components']:
                if 'delivery_ratio' in comp_dict:
                    delivery_ratios.append(comp_dict['delivery_ratio'] * 100)
                if 'avg_utilization' in comp_dict:
                    avg_utils.append(comp_dict['avg_utilization'] * 100)
            
            if delivery_ratios:
                ax.plot(epochs, delivery_ratios, 'g-', label='Доставка (%)', linewidth=2)
            if avg_utils:
                ax.plot(epochs, avg_utils, 'orange', label='Загрузка рёбер (%)', linewidth=2)
            
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Проценты')
            ax.set_title('Метрики качества')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 105])
        
        plt.tight_layout()
        
        # Сохраняем
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Графики обучения сохранены в {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_learning_curves_comparison(self,
                                        histories: List[Dict],
                                        labels: List[str],
                                        save_name: str = 'learning_curves_comparison.png',
                                        show: bool = True) -> None:
        """
        Сравнивает кривые обучения для разных запусков.
        Полезно при подборе гиперпараметров.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Train loss
        ax = axes[0]
        for history, label in zip(histories, labels):
            epochs = range(len(history['train_loss']))
            ax.plot(epochs, history['train_loss'], label=label, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Loss')
        ax.set_title('Train Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation loss
        ax = axes[1]
        for history, label in zip(histories, labels):
            epochs = range(len(history['val_loss']))
            ax.plot(epochs, history['val_loss'], label=label, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


class FlowVisualizer:
    """
    Визуализация результатов потокораспределения на графе.
    """
    
    def __init__(self, graph, feature_extractor, registry, save_dir: str = 'genereted'):
        self.graph = graph
        self.extractor = feature_extractor
        self.registry = registry
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def create_flow_html(self,
                        results: Dict,
                        flows: Dict,
                        save_name: str = 'flow_visualization.html',
                        title: str = 'Распределение потоков в сети «Альфа»') -> str:
        """
        Создаёт интерактивную HTML визуализацию с потоками.
        
        Args:
            results: результаты предсказания из FlowPredictor.predict()
            flows: исходные заявки
            save_name: имя файла для сохранения
            title: заголовок визуализации
            
        Returns:
            путь к сохранённому файлу
        """
        from pyvis.network import Network
        
        net = Network(height="800px", width="100%", bgcolor="#ffffff")
        
        # Настройки физики
        net.set_options("""
        {
        "nodes": {
            "font": {"size": 14}
        },
        "edges": {
            "color": {"inherit": false},
            "smooth": {"type": "continuous"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
        },
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 95,
                "springConstant": 0.04
            },
            "minVelocity": 0.75
        }
        }
        """)
        
        edge_flows = results['edge_flows']
        edge_utils = results['edge_utilization']
        capacities = self.extractor.get_edge_capacities()
        
        # Цвета для узлов
        color_map = {
            'source': '#e74c3c',
            'consumer': '#2ecc71',
            'junction': '#3498db',
            'additional': '#f39c12',
            'unknown': '#95a5a6'
        }
        
        shape_map = {
            'source': 'triangle',
            'consumer': 'square',
            'junction': 'dot',
            'additional': 'diamond',
            'unknown': 'dot'
        }
        
        # Добавляем узлы
        for node in self.graph.nodes.values():
            # Дополнительная информация для источников и потребителей
            title_parts = [f"Тип: {node.type}"]
            
            if node.type == 'source':
                total_gen = sum(flows.get(node.name, {}).values())
                title_parts.append(f"Генерация: {total_gen:.1f} кВт")
            elif node.type == 'consumer':
                total_cons = sum(f.get(node.name, 0) for f in flows.values())
                title_parts.append(f"Потребление: {total_cons:.1f} кВт")
            
            net.add_node(
                node.name,
                label=node.name,
                color=color_map[node.type],
                title="<br>".join(title_parts),
                shape=shape_map[node.type],
                size=25 if node.type in ['source', 'consumer'] else 15
            )
        
        # Добавляем рёбра с информацией о потоках
        max_flow = edge_flows.max() if edge_flows.max() > 0 else 1.0
        
        for i, edge in enumerate(self.extractor.edges):
            u, v = edge.nodes[0].name, edge.nodes[1].name
            cap = capacities[i]
            flow = edge_flows[i]
            util = edge_utils[i]
            
            # Цвет ребра зависит от загрузки
            if util > 0.95:
                color = '#e74c3c'  # красный - перегрузка
                width = 5
            elif util > 0.7:
                color = '#f39c12'  # оранжевый - высокая загрузка
                width = 4
            elif util > 0.3:
                color = '#f1c40f'  # жёлтый - средняя загрузка
                width = 3
            else:
                color = '#2ecc71'  # зелёный - низкая загрузка
                width = 2
            
            # Подпись при наведении
            if cap >= 1e8:
                cap_str = "∞"
                util_str = "—"
            else:
                cap_str = f"{cap:.1f}"
                util_str = f"{util*100:.1f}%"
            
            title = (f"Поток: {flow:.1f} / {cap_str} кВт<br>"
                    f"Загрузка: {util_str}")
            
            net.add_edge(u, v, title=title, color=color, width=width, 
                        arrows='to')
        
        # Генерируем HTML
        html_content = net.generate_html()
        
        # Добавляем легенду и статистику
        total_demanded = results.get('demanded', 0)
        total_delivered = results.get('total_delivered', 0)
        delivery_pct = (total_delivered / total_demanded * 100) if total_demanded > 0 else 0
        
        # Находим перегруженные рёбра
        overloaded = np.where(edge_utils > 0.95)[0]
        high_load = np.where((edge_utils > 0.7) & (edge_utils <= 0.95))[0]
        
        legend_html = f"""
        <div style="position: absolute; top: 10px; right: 10px; background: white; 
                    padding: 15px; border: 1px solid #ccc; border-radius: 8px; 
                    z-index: 1000; font-family: Arial, sans-serif; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <b style="font-size: 16px;">{title}</b><br><br>
            
            <b>Статистика:</b><br>
            Заявлено: {total_demanded:.1f} кВт<br>
            Доставлено: {total_delivered:.1f} кВт<br>
            Доставка: {delivery_pct:.1f}%<br><br>
            
            <b>Загрузка рёбер:</b><br>
            <span style="color: #e74c3c;">■ Перегрузка (>95%)</span>: {len(overloaded)}<br>
            <span style="color: #f39c12;">■ Высокая (>70%)</span>: {len(high_load)}<br>
            <span style="color: #f1c40f;">■ Средняя (30-70%)</span><br>
            <span style="color: #2ecc71;">■ Низкая (<30%)</span><br><br>
            
            <b>Легенда узлов:</b><br>
            <span style="color: #e74c3c;">▲ Источники</span><br>
            <span style="color: #2ecc71;">■ Потребители</span><br>
            <span style="color: #3498db;">● Узлы</span><br>
            <span style="color: #f39c12;">◆ Вспомогательные</span>
        </div>
        """
        
        html_content = html_content.replace('</body>', f'{legend_html}</body>')
        
        save_path = os.path.join(self.save_dir, save_name)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Визуализация потоков сохранена в {save_path}")
        return save_path
    
    def plot_sensitivity_analysis(self,
                                predictor,
                                base_flows: Dict,
                                parameter: str = 'demand',
                                variation_range: Tuple[float, float] = (0.5, 1.5),
                                num_points: int = 20,
                                save_name: str = 'sensitivity.png',
                                show: bool = True) -> Dict:
        """
        Анализ чувствительности: как меняется доставка при изменении заявок.
        
        Args:
            predictor: FlowPredictor
            base_flows: базовые заявки
            parameter: что варьировать ('demand', 'capacity')
            variation_range: диапазон варьирования (множитель)
            num_points: количество точек
            save_name: имя файла для сохранения
            show: показать ли график
            
        Returns:
            словарь с результатами анализа
        """
        import copy
        
        multipliers = np.linspace(variation_range[0], variation_range[1], num_points)
        
        delivery_ratios = []
        overload_counts = []
        avg_utils = []
        
        for mult in multipliers:
            # Создаём модифицированные заявки
            modified_flows = copy.deepcopy(base_flows)
            
            if parameter == 'demand':
                for source in modified_flows:
                    for consumer in modified_flows[source]:
                        modified_flows[source][consumer] *= mult
            
            # Предсказываем
            results = predictor.predict(modified_flows)
            
            # Собираем метрики
            ratio = results['total_delivered'] / results['demanded'] if results['demanded'] > 0 else 0
            delivery_ratios.append(ratio * 100)
            
            overloaded = np.sum(results['edge_utilization'] > 0.95)
            overload_counts.append(overloaded)
            
            avg_utils.append(np.mean(results['edge_utilization']) * 100)
        
        # Строим графики
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Доставка
        ax = axes[0]
        ax.plot(multipliers, delivery_ratios, 'b-o', linewidth=2, markersize=6)
        ax.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='100%')
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Базовый уровень')
        ax.set_xlabel('Множитель заявок')
        ax.set_ylabel('Доставка (%)')
        ax.set_title('Зависимость доставки от объёма заявок')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Перегруженные рёбра
        ax = axes[1]
        ax.plot(multipliers, overload_counts, 'r-o', linewidth=2, markersize=6)
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Множитель заявок')
        ax.set_ylabel('Количество перегруженных рёбер')
        ax.set_title('Зависимость перегрузок от объёма заявок')
        ax.grid(True, alpha=0.3)
        
        # Средняя загрузка
        ax = axes[2]
        ax.plot(multipliers, avg_utils, 'orange', linestyle='-', marker='o', linewidth=2, markersize=6)
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Порог 70%')
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Множитель заявок')
        ax.set_ylabel('Средняя загрузка рёбер (%)')
        ax.set_title('Зависимость средней загрузки от объёма заявок')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return {
            'multipliers': multipliers.tolist(),
            'delivery_ratios': delivery_ratios,
            'overload_counts': overload_counts,
            'avg_utils': avg_utils
        }
    
    def save_results_report(self,
                           results: Dict,
                           flows: Dict,
                           save_name: str = 'results_report.json') -> None:
        """
        Сохраняет подробный отчёт о результатах в JSON.
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_demanded': float(results['demanded']),
                'total_delivered': float(results['total_delivered']),
                'delivery_ratio': float(results['total_delivered'] / results['demanded'] if results['demanded'] > 0 else 0),
            },
            'edges': []
        }
        
        # Информация о рёбрах
        capacities = self.extractor.get_edge_capacities()
        edge_flows = results['edge_flows']
        edge_utils = results['edge_utilization']
        
        for i, edge in enumerate(self.extractor.edges):
            cap = capacities[i]
            report['edges'].append({
                'from': edge.nodes[0].name,
                'to': edge.nodes[1].name,
                'capacity': float('inf') if cap >= 1e8 else float(cap),
                'flow': float(edge_flows[i]),
                'utilization': float(edge_utils[i]),
                'is_overloaded': bool(edge_utils[i] > 0.95)
            })
        
        # Информация о заявках
        report['requests'] = []
        for source, consumers in flows.items():
            for consumer, demand in consumers.items():
                report['requests'].append({
                    'source': source,
                    'consumer': consumer,
                    'demanded': float(demand),
                })
        
        save_path = os.path.join(self.save_dir, save_name)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Отчёт сохранён в {save_path}")
