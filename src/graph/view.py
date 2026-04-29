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

    def draw_with_loads(self, edge_loads, filename="solution_graph.html"):
        """
        Визуализация с отображением потоков и загрузки рёбер.
        edge_loads: словарь {Edge: (flow, load_ratio)}
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
            "arrows": {"to": {"enabled": true}}
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

        for node in self.graph.nodes.values():
            net.add_node(node.name, label=node.name,
                         color=color_map[node.type],
                         title=f"Тип: {node.type}",
                         shape=shape_map[node.type],
                         size=25 if node.type in ['source', 'consumer'] else 15)

        max_flow = max([flow for flow, _ in edge_loads.values()], default=1.0)

        for edge, (flow, ratio) in edge_loads.items():
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

            title = f"Flow: {flow:.2f} / {cap} kW<br>Load: {ratio*100:.1f}%"
            net.add_edge(u, v, title=title, color=color, width=width, arrows='to')

        html = net.generate_html()
        legend = f"""
        <div style="position: absolute; top: 10px; right: 10px; background: white;
                    padding: 15px; border: 1px solid #ccc; border-radius: 8px;
                    z-index: 1000; font-family: Arial; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <b>Legend</b><br>
            <span style="color:#e74c3c;">▲ Source</span><br>
            <span style="color:#2ecc71;">■ Consumer</span><br>
            <span style="color:#3498db;">● Junction</span><br>
            <span style="color:#f39c12;">◆ Additional</span><br>
            <hr>
            <span style="color:#e74c3c;">Red</span> &gt;95%<br>
            <span style="color:#f39c12;">Orange</span> 70-95%<br>
            <span style="color:#f1c40f;">Yellow</span> 30-70%<br>
            <span style="color:#2ecc71;">Green</span> &lt;30%
        </div>
        """
        html = html.replace('</body>', f'{legend}</body>')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Граф решения сохранён в {filename}")
