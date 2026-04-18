"""Модуль для визуализации графа электрической сети."""

import os
from pyvis.network import Network


class GraphView:
    """Класс для визуализации графа."""
    
    def __init__(self, graph):
        self.graph = graph

    def draw_pyvis(self, filename="graph.html"):
        """Интерактивная визуализация с PyVis."""
        
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