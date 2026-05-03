import os
from pyvis.network import Network

class GraphView:
    """Класс для визуализации графа"""
    def __init__(self, graph):
        self.graph = graph

    def draw_pyvis(self, filename="graph.html"):
        """Интерактивная визуализация с PyVis"""
        net = Network(height="800px", width="100%", bgcolor="#ffffff")
        
        # Настройки физики для лучшего отображения
        net.set_options("""
        {
        "nodes": {
            "font": {
            "size": 14
            }
        },
        "edges": {
            "color": {
            "inherit": true
            },
            "smooth": {
            "type": "continuous"
            }
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

        color_map = {
            'source': '#e74c3c',      # красный
            'consumer': '#2ecc71',    # зелёный
            'junction': '#3498db',    # синий
            'additional': '#f39c12',  # оранжевый
            'unknown': '#95a5a6'      # серый
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
            net.add_node(
                node.name,
                label=node.name,
                color=color_map[node.type],
                title=f"Тип: {node.type}",
                shape=shape_map[node.type],
                size=20 if node.type in ['source', 'consumer'] else 15
            )

        # Добавляем рёбра
        for edge in self.graph.edges:
            u, v = edge.nodes[0].name, edge.nodes[1].name
            cap = edge.capacity if edge.capacity != float('inf') else '∞'
            net.add_edge(
                u, v,
                title=f"Пропускная способность: {cap} кВт",
                label=str(cap) if edge.capacity != float('inf') else "∞",
                color="#888888",
                width=2
            )

        # Сохраняем HTML напрямую, без использования show()
        html_content = net.generate_html()
        
        # Добавляем легенду
        legend_html = """
        <div style="position: absolute; top: 10px; right: 10px; background: white; 
                    padding: 10px; border: 1px solid #ccc; border-radius: 5px; z-index: 1000;">
            <b>Легенда:</b><br>
            <span style="color: #e74c3c;">▲ Источники</span><br>
            <span style="color: #2ecc71;">■ Потребители</span><br>
            <span style="color: #3498db;">● Узлы</span><br>
            <span style="color: #f39c12;">◆ Вспомогательные</span>
        </div>
        """
        
        # Вставляем легенду перед закрывающим тегом body
        html_content = html_content.replace('</body>', f'{legend_html}</body>')
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Граф сохранён в {filename}")
        print(f"Откройте файл в браузере: file://{os.path.abspath(filename)}")

    def draw_with_directed_flows(self, edge_loads, directed_flows, 
                                 filename="solution-graph.html",
                                 title="Распределение потоков",
                                 total_demanded=None,
                                 total_delivered=None,
                                 from_ml=False):
        """
        Визуализация с направленными потоками
        
        Args:
            edge_loads: {Edge: (total_flow, load_ratio)} — суммарные потоки и загрузка
            directed_flows: {(from_node, to_node): flow} — направленные потоки
            filename: имя выходного файла
        """
        net = Network(height="800px", width="100%", bgcolor="#ffffff")

        net.set_options("""
        {
        "nodes": {
            "font": {"size": 14}
        },
        "edges": {
            "color": {"inherit": false},
            "smooth": {"type": "continuous"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}}
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
            net.add_node(
                node.name,
                label=node.name,
                color=color_map[node.type],
                title=f"Тип: {node.type}",
                shape=shape_map[node.type],
                size=25 if node.type in ['source', 'consumer'] else 15
            )

        # Добавляем рёбра
        for edge, (total_flow, ratio) in edge_loads.items():
            u = edge.nodes[0].name
            v = edge.nodes[1].name
            cap = edge.capacity if edge.capacity != float('inf') else float('inf')
            
            # Направленные потоки
            flow_uv = directed_flows.get((u, v), 0.0)
            flow_vu = directed_flows.get((v, u), 0.0)
            
            # Цвет по загрузке
            if ratio > 1.0:
                color = '#e74c3c'
            elif ratio > 0.8:
                color = '#f39c12'
            elif ratio > 0.6:
                color = '#f1c40f'
            else:
                color = '#2ecc71'
            
            width = max(2, ratio * 5)
            
            # Форматируем capacity
            if cap == float('inf'):
                cap_str = "∞"
            else:
                cap_str = f"{cap:.1f}"
            
            # Подсказка при наведении
            title = (
                f"Ребро: {u} ↔ {v}\n"
                f"────────────────\n"
                f"Поток {u} → {v}: {flow_uv:.2f} кВт\n"
                f"Поток {v} → {u}: {flow_vu:.2f} кВт\n"
                f"────────────────\n"
                f"Суммарный поток по ребру: {total_flow:.2f} кВт\n"
                f"Максимальная вместимость: {cap_str} кВт\n"
                f"Загрузка: {ratio*100:.1f}%"
            )
            
            # Если поток течёт в обе стороны — добавляем ДВА ребра
            if flow_uv > 0 and flow_vu > 0:
                # Доминирующее направление (основное ребро)
                if flow_uv >= flow_vu:
                    net.add_edge(u, v, title=title, color=color, width=width, arrows='to')
                    # Обратное направление (тонкая линия)
                    # back_width = max(1, (flow_vu / max(flow_uv, 1)) * width)
                    # net.add_edge(v, u, title=title, color='#95a5a6', width=back_width, arrows='to')
                else:
                    net.add_edge(v, u, title=title, color=color, width=width, arrows='to')
                    # back_width = max(1, (flow_uv / max(flow_vu, 1)) * width)
                    # net.add_edge(u, v, title=title, color='#95a5a6', width=back_width, arrows='to')
            else:
                # Поток только в одну сторону
                if flow_uv > 0:
                    net.add_edge(u, v, title=title, color=color, width=width, arrows='to')
                else:
                    net.add_edge(v, u, title=title, color=color, width=width, arrows='to')

        # Генерируем HTML
        html = net.generate_html()

        # Статистика
        stats_html = ""
        if total_demanded is not None and total_delivered is not None:
            delivery_pct = (total_delivered / total_demanded * 100) if total_demanded > 0 else 0
            stats_html = f"""
            <b>Статистика:</b><br>
            Заявлено: {total_demanded:.1f} кВт<br>
            Доставлено: {total_delivered:.1f} кВт<br>
            Доставка: {delivery_pct:.1f}%<br><br>
            """
        
        # Подсчёт рёбер по типам загрузки
        overloaded = sum(1 for _, ratio in edge_loads.values() if ratio > 1.0)
        high_load = sum(1 for _, ratio in edge_loads.values() if 0.8 < ratio <= 1.0)
        medium_load = sum(1 for _, ratio in edge_loads.values() if 0.6 < ratio <= 0.8)
        low_load = sum(1 for _, ratio in edge_loads.values() if ratio <= 0.6)
        
                # Генерируем HTML
        html = net.generate_html()

        # Вставляем стили в head для растягивания на весь экран
        style_tag = """
        <style>
            html, body {
                margin: 0 !important;
                padding: 0 !important;
                width: 100% !important;
                height: 100% !important;
                overflow: hidden !important;
            }
            #mynetwork {
                width: 100% !important;
                height: 100vh !important;
            }
        </style>
        """
        html = html.replace('</head>', f'{style_tag}</head>')

        # Статистика
        stats_html = ""
        if total_demanded is not None and total_delivered is not None:
            delivery_pct = (total_delivered / total_demanded * 100) if total_demanded > 0 else 0
            stats_html = f"""
            <b>Статистика:</b><br>
            Заявлено: {total_demanded:.1f} кВт<br>
            Доставлено: {total_delivered:.1f} кВт<br>
            Доставка: {delivery_pct:.1f}%<br><br>
            """
        
        # Подсчёт рёбер по типам загрузки
        overloaded = sum(1 for _, ratio in edge_loads.values() if ratio > 1.0)
        high_load = sum(1 for _, ratio in edge_loads.values() if 0.8 < ratio <= 1.0)
        medium_load = sum(1 for _, ratio in edge_loads.values() if 0.6 < ratio <= 0.8)
        low_load = sum(1 for _, ratio in edge_loads.values() if ratio <= 0.6)
        
        legend = f"""
        <div style="position: absolute; top: 10px; right: 10px; background: white;
                    padding: 15px; border: 1px solid #ccc; border-radius: 8px;
                    z-index: 1000; font-family: Arial, sans-serif; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 250px;">
            
            {stats_html}
            
            <b>Загрузка рёбер:</b><br>
            <span style="color:#e74c3c;">■ Критическая >100%</span>: {overloaded}<br>
            <span style="color:#f39c12;">■ Высокая 80-100%</span>: {high_load}<br>
            <span style="color:#f1c40f;">■ Средняя 60-80%</span>: {medium_load}<br>
            <span style="color:#2ecc71;">■ Низкая <60%</span>: {low_load}<br><br>
            
            <b>Типы узлов:</b><br>
            <span style="color:#e74c3c;">▲ Источник</span><br>
            <span style="color:#2ecc71;">■ Потребитель</span><br>
            <span style="color:#3498db;">● Узел соединения</span><br>
            <span style="color:#f39c12;">◆ Вспомогательный</span><br><br>
            
            <small>
            • Стрелка → направление потока<br>
            • Толщина линии ∝ величине потока<br>
            • При наведении — детали потоков
            </small>
        </div>
        """
        html = html.replace('</body>', f'{legend}</body>')

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f" Граф с направленными потоками сохранён в {filename}")