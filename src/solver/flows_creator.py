"""Загрузка ненулевых потоков из flows.json и создание FlowInstance."""

import json
from typing import List, Dict, Optional
from graph import Graph, Request, RequestRegistry
from .flow_instance import FlowInstance


class FlowsCreator:
    """Создаёт FlowInstance на основе flows.json и построенных путей."""
    
    def __init__(self, graph: Graph, registry: RequestRegistry):
        self.graph = graph
        self.registry = registry
    
    def create_from_file(self, flows_file: str) -> List[FlowInstance]:
        """
        Загружает flows.json и создаёт FlowInstance только для ненулевых потоков.
        """
        with open(flows_file, 'r', encoding='utf-8') as f:
            flows_data = json.load(f)
        
        # Строим индекс заявок по паре (source_name, consumer_name)
        request_index: Dict[tuple, Request] = {}
        for req in self.registry.requests:
            key = (req.source.name, req.consumer.name)
            request_index[key] = req
        
        instances = []
        for source_name, consumers in flows_data.items():
            source_node = self.graph.get_node(source_name)
            if source_node is None:
                print(f"Предупреждение: источник {source_name} не найден в графе")
                continue
            
            for consumer_name, amount_str in consumers.items():
                amount = float(amount_str)
                if amount <= 0:
                    continue  # пропускаем нулевые
                
                consumer_node = self.graph.get_node(consumer_name)
                if consumer_node is None:
                    print(f"Предупреждение: потребитель {consumer_name} не найден в графе")
                    continue
                
                key = (source_name, consumer_name)
                request = request_index.get(key)
                if request is None:
                    print(f"Предупреждение: заявка {source_name} -> {consumer_name} не найдена в реестре (возможно, нет путей)")
                    continue
                
                instance = FlowInstance(request, amount)
                instances.append(instance)
        
        return instances