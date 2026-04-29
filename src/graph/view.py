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

    def draw_with_loads(self, edge_loads, actual_flows_dict=None, filename="solution_graph.html"):
        """
        Визуализация с отображением направленных потоков.
        
        Args:
            edge_loads: словарь {Edge: (total_flow, load_ratio)} — суммарные потоки
            actual_flows_dict: словарь {(inst_idx, path_key): flow} — детальные потоки по путям
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

        # Если есть детальные потоки — вычисляем направленные потоки по рёбрам
        if actual_flows_dict:
            # Строим направленные потоки: (u, v) -> flow
            directed_flows = {}
            for (inst_idx, path_key), flow in actual_flows_dict.items():
                # Нам нужен доступ к экземплярам, но здесь их нет.
                # Поэтому используем простой подход: суммируем абсолютные значения
                pass  # Будет реализовано при передаче instances

            # Для каждого ребра проверяем направления
            for edge, (total_flow, ratio) in edge_loads.items():
                u, v = edge.nodes[0].name, edge.nodes[1].name
                cap = edge.capacity if edge.capacity != float('inf') else float('inf')

                # Определяем цвет и толщину по загрузке
                if ratio > 0.95:
                    color = '#e74c3c'; width = 5
                elif ratio > 0.7:
                    color = '#f39c12'; width = 4
                elif ratio > 0.3:
                    color = '#f1c40f'; width = 3
                else:
                    color = '#2ecc71'; width = 2

                # Добавляем ОДНО ребро с суммарным потоком и стрелкой
                # Если есть направленные данные - показываем их
                title = (f"Суммарный поток: {total_flow:.2f} / {cap} кВт<br>"
                        f"Загрузка: {ratio*100:.1f}%")
                
                net.add_edge(u, v, title=title, color=color, width=width, arrows='to')

        else:
            # Без детальных данных - просто показываем суммарные потоки
            for edge, (total_flow, ratio) in edge_loads.items():
                u, v = edge.nodes[0].name, edge.nodes[1].name
                cap = edge.capacity if edge.capacity != float('inf') else float('inf')

                if ratio > 0.95:
                    color = '#e74c3c'; width = 5
                elif ratio > 0.7:
                    color = '#f39c12'; width = 4
                elif ratio > 0.3:
                    color = '#f1c40f'; width = 3
                else:
                    color = '#2ecc71'; width = 2

                title = (f"Поток: {total_flow:.2f} / {cap} кВт<br>"
                        f"Загрузка: {ratio*100:.1f}%")
                
                net.add_edge(u, v, title=title, color=color, width=width, arrows='to')

        # Генерируем HTML
        html = net.generate_html()

        # Легенда
        legend = f"""
        <div style="position: absolute; top: 10px; right: 10px; background: white;
                    padding: 15px; border: 1px solid #ccc; border-radius: 8px;
                    z-index: 1000; font-family: Arial; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <b>Легенда</b><br>
            <span style="color:#e74c3c;">▲ Источник</span><br>
            <span style="color:#2ecc71;">■ Потребитель</span><br>
            <span style="color:#3498db;">● Узел</span><br>
            <span style="color:#f39c12;">◆ Вспомогательный</span>
            <hr>
            <b>Загрузка рёбер:</b><br>
            <span style="color:#e74c3c;">■ Перегрузка >95%</span><br>
            <span style="color:#f39c12;">■ Высокая 70-95%</span><br>
            <span style="color:#f1c40f;">■ Средняя 30-70%</span><br>
            <span style="color:#2ecc71;">■ Низкая <30%</span>
            <hr>
            <small>Стрелка → направление суммарного потока<br>
            При наведении — детальная информация</small>
        </div>
        """
        html = html.replace('</body>', f'{legend}</body>')

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f" Граф решения сохранён в {filename}")

    def draw_with_directed_flows(self, edge_loads, directed_flows, filename="solution_graph.html"):
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

        # Добавляем рёбра с направленными потоками
        for edge, (total_flow, ratio) in edge_loads.items():
            u, v = edge.nodes[0].name, edge.nodes[1].name
            cap = edge.capacity if edge.capacity != float('inf') else float('inf')
            
            # Направленные потоки
            flow_uv = directed_flows.get((u, v), 0.0)
            flow_vu = directed_flows.get((v, u), 0.0)
            
            # Определяем доминирующее направление и чистый поток
            if flow_uv >= flow_vu:
                direction_text = f"{u} → {v}"
                net_flow = flow_uv - flow_vu
            else:
                direction_text = f"{v} → {u}"
                net_flow = flow_vu - flow_uv
            
            # Цвет по загрузке
            if ratio > 0.95:
                color = '#e74c3c'
            elif ratio > 0.7:
                color = '#f39c12'
            elif ratio > 0.3:
                color = '#f1c40f'
            else:
                color = '#2ecc71'
            
            # Толщина ребра пропорциональна загрузке
            width = max(2, ratio * 5)
            
            # Подпись при наведении
            if cap == float('inf'):
                cap_str = "∞"
            else:
                cap_str = f"{cap:.1f}"
            
            title = (f"<b>Ребро: {u} ↔ {v}</b><br>"
                    f"Доминирующее направление: {direction_text}<br>"
                    f"Чистый поток: {net_flow:.2f} кВт<br>"
                    f"<hr>"
                    f"Поток {u} → {v}: {flow_uv:.2f} кВт<br>"
                    f"Поток {v} → {u}: {flow_vu:.2f} кВт<br>"
                    f"<hr>"
                    f"Суммарный поток: {total_flow:.2f} / {cap_str} кВт<br>"
                    f"Загрузка: {ratio*100:.1f}%")
            
            net.add_edge(u, v, title=title, color=color, width=width, arrows='to')

        # Генерируем HTML
        html = net.generate_html()

        # Легенда
        legend = f"""
        <div style="position: absolute; top: 10px; right: 10px; background: white;
                    padding: 15px; border: 1px solid #ccc; border-radius: 8px;
                    z-index: 1000; font-family: Arial, sans-serif; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 250px;">
            <b style="font-size: 14px;">Легенда</b><br><br>
            
            <b>Типы узлов:</b><br>
            <span style="color:#e74c3c;">▲ Источник</span><br>
            <span style="color:#2ecc71;">■ Потребитель</span><br>
            <span style="color:#3498db;">● Узел соединения</span><br>
            <span style="color:#f39c12;">◆ Вспомогательный</span><br><br>
            
            <b>Загрузка рёбер:</b><br>
            <span style="color:#e74c3c;">■ Критическая >95%</span><br>
            <span style="color:#f39c12;">■ Высокая 70-95%</span><br>
            <span style="color:#f1c40f;">■ Средняя 30-70%</span><br>
            <span style="color:#2ecc71;">■ Низкая <30%</span><br><br>
            
            <small>
            • Стрелка → доминирующее направление<br>
            • Толщина линии ∝ загрузке<br>
            • При наведении — детали потоков
            </small>
        </div>
        """
        html = html.replace('</body>', f'{legend}</body>')

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✓ Граф с направленными потоками сохранён в {filename}")

