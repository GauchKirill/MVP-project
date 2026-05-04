"""Модуль для визуализации графа электрической сети."""

import os
from pyvis.network import Network
from collections import defaultdict


class GraphView:
    """Класс для визуализации графа"""
    
    def __init__(self, graph):
        self.graph = graph

    def draw_pyvis(self, filename="graph.html"):
        """Интерактивная визуализация с PyVis (без потоков)."""
        net = Network(height="800px", width="100%", bgcolor="#ffffff")
        
        net.set_options("""
        {
        "nodes": {"font": {"size": 14}},
        "edges": {"color": {"inherit": true}, "smooth": {"type": "continuous"}},
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -8000, "centralGravity": 0.3,
                "springLength": 95, "springConstant": 0.04
            },
            "minVelocity": 0.75
        }
        }
        """)

        color_map = {
            'source': '#e74c3c', 'consumer': '#2ecc71',
            'junction': '#3498db', 'additional': '#f39c12', 'unknown': '#95a5a6'
        }
        shape_map = {
            'source': 'triangle', 'consumer': 'square',
            'junction': 'dot', 'additional': 'diamond', 'unknown': 'dot'
        }

        for node in self.graph.nodes.values():
            net.add_node(
                node.name, label=node.name,
                color=color_map[node.type],
                title=f"Тип: {node.type}",
                shape=shape_map[node.type],
                size=20 if node.type in ['source', 'consumer'] else 15
            )

        for edge in self.graph.edges:
            u, v = edge.nodes[0].name, edge.nodes[1].name
            cap = edge.capacity if edge.capacity != float('inf') else '∞'
            net.add_edge(u, v,
                        title=f"Пропускная способность: {cap} кВт",
                        label=str(cap) if edge.capacity != float('inf') else "∞",
                        color="#888888", width=2)

        html_content = net.generate_html()
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
        html_content = html_content.replace('</body>', f'{legend_html}</body>')
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Граф сохранён в {filename}")

    def draw_with_directed_flows(self, edge_loads, directed_flows, 
                             filename="solution-graph.html",
                             title="Распределение потоков",
                             total_demanded=None,
                             total_delivered=None,
                             flows_data=None,
                             source_delivery=None,
                             consumer_receipt=None):
        """
        Визуализация с направленными потоками.
        
        Args:
            edge_loads: {Edge: (total_flow, load_ratio)}
            directed_flows: {(from_node, to_node): flow}
            flows_data: словарь {source: {consumer: demand}} — заявки
            source_delivery: {source: {consumer: delivered}} — факт доставки от источника к потребителю
            consumer_receipt: {consumer: {source: received}} — факт получения потребителем
        """
        net = Network(height="800px", width="100%", bgcolor="#ffffff")

        net.set_options("""
        {
        "nodes": {"font": {"size": 14}},
        "edges": {
            "color": {"inherit": false},
            "smooth": {"type": "continuous"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}}
        },
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -8000, "centralGravity": 0.3,
                "springLength": 95, "springConstant": 0.04
            },
            "minVelocity": 0.75
        }
        }
        """)

        color_map = {
            'source': '#e74c3c', 'consumer': '#2ecc71',
            'junction': '#3498db', 'additional': '#f39c12', 'unknown': '#95a5a6'
        }
        shape_map = {
            'source': 'triangle', 'consumer': 'square',
            'junction': 'dot', 'additional': 'diamond', 'unknown': 'dot'
        }

        # Используем переданные фактические доставки
        source_actual = defaultdict(lambda: defaultdict(float))
        consumer_actual = defaultdict(lambda: defaultdict(float))
        
        if source_delivery:
            for src, consumers in source_delivery.items():
                for cons, flow in consumers.items():
                    source_actual[src][cons] = flow
        
        if consumer_receipt:
            for cons, sources in consumer_receipt.items():
                for src, flow in sources.items():
                    consumer_actual[cons][src] = flow

        # ===== Добавляем узлы =====
        for node in self.graph.nodes.values():
            title_parts = [f"Узел: {node.name}", f"Тип: {node.type}"]
            
            if node.type == 'source' and flows_data and node.name in flows_data:
                title_parts.append("")
                title_parts.append("Доставка потребителям:")
                
                total_demanded_src = 0
                total_delivered_src = 0
                
                for consumer, demand in sorted(flows_data[node.name].items()):
                    delivered = source_actual.get(node.name, {}).get(consumer, 0.0)
                    pct = (delivered / demand * 100) if demand > 0 else 0
                    title_parts.append(
                        f"  {consumer}: {delivered:.1f} / {demand:.1f} | {pct:.1f}%"
                    )
                    total_demanded_src += demand
                    total_delivered_src += delivered
                
                if total_demanded_src > 0:
                    total_pct = (total_delivered_src / total_demanded_src * 100)
                    title_parts.append("  -----------------------------")
                    title_parts.append(
                        f"  ИТОГО: {total_delivered_src:.1f} / {total_demanded_src:.1f} | {total_pct:.1f}%"
                    )
            
            if node.type == 'consumer' and flows_data:
                # Собираем заявки к этому потребителю
                source_demands = {}
                for source, consumers in flows_data.items():
                    if node.name in consumers:
                        source_demands[source] = consumers[node.name]
                
                if source_demands:
                    title_parts.append("")
                    title_parts.append("Получено от источников:")
                    
                    total_demanded_cons = 0
                    total_delivered_cons = 0
                    
                    for source, demand in sorted(source_demands.items()):
                        delivered = consumer_actual.get(node.name, {}).get(source, 0.0)
                        pct = (delivered / demand * 100) if demand > 0 else 0
                        title_parts.append(
                            f"  {source}: {delivered:.1f} / {demand:.1f} | {pct:.1f}%"
                        )
                        total_demanded_cons += demand
                        total_delivered_cons += delivered
                    
                    if total_demanded_cons > 0:
                        total_pct = (total_delivered_cons / total_demanded_cons * 100)
                        title_parts.append("  -----------------------------")
                        title_parts.append(
                            f"  ИТОГО: {total_delivered_cons:.1f} / {total_demanded_cons:.1f} | {total_pct:.1f}%"
                        )
            
            title_text = "\n".join(title_parts)
            
            net.add_node(
                node.name, label=node.name,
                color=color_map[node.type], title=title_text,
                shape=shape_map[node.type],
                size=30 if node.type in ['source', 'consumer'] else 15
            )

        # ===== Добавляем рёбра =====
        for edge, (total_flow, ratio) in edge_loads.items():
            u = edge.nodes[0].name
            v = edge.nodes[1].name
            cap = edge.capacity if edge.capacity != float('inf') else float('inf')
            
            flow_uv = float(directed_flows.get((u, v), 0.0))
            flow_vu = float(directed_flows.get((v, u), 0.0))
            total_flow = float(total_flow)
            ratio = float(ratio)
            
            if flow_uv >= flow_vu:
                direction = f"{u} → {v}"
                from_node, to_node = u, v
            else:
                direction = f"{v} → {u}"
                from_node, to_node = v, u
            
            if ratio > 1.0:
                color = '#8B0000'
            elif ratio > 0.95:
                color = '#e74c3c'
            elif ratio > 0.8:
                color = '#f39c12'
            elif ratio > 0.6:
                color = '#f1c40f'
            else:
                color = '#2ecc71'
            
            width = max(2, min(ratio * 8, 10))
            cap_str = "∞" if cap == float('inf') else f"{cap:.1f}"
            
            edge_title = (
                f"Ребро: {u} ↔ {v}\n"
                f"Направление: {direction}\n"
                f"────────────────────────\n"
                f"Поток {u}→{v}: {flow_uv:.2f} кВт\n"
                f"Поток {v}→{u}: {flow_vu:.2f} кВт\n"
                f"────────────────────────\n"
                f"Суммарный поток: {total_flow:.2f} кВт\n"
                f"Максимум: {cap_str} кВт\n"
                f"Загрузка: {ratio*100:.1f}%"
            )
            
            if abs(flow_uv - flow_vu) > 0.01:
                net.add_edge(from_node, to_node, title=edge_title, color=color, width=width, arrows='to')
            elif total_flow > 0.01:
                net.add_edge(u, v, title=edge_title, color=color, width=width)
            else:
                net.add_edge(u, v, title=edge_title, color='#d5dbdb', width=1)

        html = net.generate_html()

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
                position: absolute !important;
                top: 0 !important;
                left: 0 !important;
            }
        </style>
        """
        html = html.replace('</head>', f'{style_tag}</head>')

        stats_html = ""
        if total_demanded is not None and total_delivered is not None:
            delivery_pct = (total_delivered / total_demanded * 100) if total_demanded > 0 else 0
            if delivery_pct > 99:
                dcolor = "#27ae60"
            elif delivery_pct < 95:
                dcolor = "#e74c3c"
            else:
                dcolor = "#f39c12"
            stats_html = f"""
            <b>Статистика:</b><br>
            Заявлено: <b>{total_demanded:,.1f} кВт</b><br>
            Доставлено: <b style="color:{dcolor};">{total_delivered:,.1f} кВт</b><br>
            Доставка: <b style="color:{dcolor};">{delivery_pct:.1f}%</b><br><br>
            """
        
        overloaded = sum(1 for _, r in edge_loads.values() if r > 1.0)
        critical_load = sum(1 for _, r in edge_loads.values() if 0.95 < r <= 1.0)
        high_load = sum(1 for _, r in edge_loads.values() if 0.8 < r <= 0.95)
        medium_load = sum(1 for _, r in edge_loads.values() if 0.6 < r <= 0.8)
        low_load = sum(1 for _, r in edge_loads.values() if r <= 0.6)
        
        legend = f"""
        <div style="position: absolute; top: 10px; right: 10px; background: white;
                    padding: 15px; border: 1px solid #ccc; border-radius: 8px;
                    z-index: 1000; font-family: Arial, sans-serif; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 280px;
                    font-size: 13px;">
            <b style="font-size: 15px;">{title}</b><br><br>
            {stats_html}
            <b>Загрузка рёбер:</b><br>
            <span style="color:#8B0000;">■ Критическое &gt;100%</span>: {overloaded}<br>
            <span style="color:#e74c3c;">■ Предельное 95-100%</span>: {critical_load}<br>
            <span style="color:#f39c12;">■ Высокое 80-95%</span>: {high_load}<br>
            <span style="color:#f1c40f;">■ Среднее 60-80%</span>: {medium_load}<br>
            <span style="color:#2ecc71;">■ Низкое &lt;60%</span>: {low_load}<br><br>
            <b>Типы узлов:</b><br>
            <span style="color:#e74c3c;">▲ Источник</span><br>
            <span style="color:#2ecc71;">■ Потребитель</span><br>
            <span style="color:#3498db;">● Узел</span><br>
            <span style="color:#f39c12;">◆ Вспомогательный</span><br><br>
            <small>
            • Стрелка → направление потока<br>
            • Толщина ∝ загрузке<br>
            • Наведите на узел — заявки и факт<br>
            • Наведите на ребро — потоки
            </small>
        </div>
        """
        html = html.replace('</body>', f'{legend}</body>')

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✓ Граф сохранён в {filename}")
