"""Модуль для управления задачей распределения потоков."""

from typing import List, Dict, Optional
from graph.model import Graph, Node, Edge
from .models import Flow
from .path_finder import PathFinder


class FlowTask:
    """Класс для хранения и управления потоками задачи."""
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.flows: List[Flow] = []
        self.path_finder = PathFinder(graph)
    
    def generate_all_pairs(self) -> int:
        """
        Генерирует потоки для всех возможных пар источник-потребитель.
        Величина потока устанавливается в 0 (неизвестна).
        """
        sources = self.graph.get_sources()
        consumers = self.graph.get_consumers()
        
        self.flows = []
        for source in sources:
            for consumer in consumers:
                flow = Flow(source, consumer, 0.0)  # величина потока не задана
                self.flows.append(flow)
        
        return len(self.flows)
    
    def build_all_paths(self, max_depth: int = 50) -> None:
        """
        Строит все возможные простые пути для каждого потока.
        
        Args:
            max_depth: максимальная глубина поиска путей
        """
        total_paths = 0
        
        for flow in self.flows:
            paths = self.path_finder.find_all_simple_paths(
                flow.source, 
                flow.consumer, 
                max_depth=max_depth
            )
            flow.paths = paths
            total_paths += len(paths)
        
        if self.flows:
            print(f"Построено путей: {total_paths} (в среднем {total_paths/len(self.flows):.1f} на поток)")
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику по задаче."""
        flows_with_paths = sum(1 for flow in self.flows if flow.paths)
        flows_without_paths = len(self.flows) - flows_with_paths
        
        return {
            'total_flows': len(self.flows),
            'total_sources': len(self.graph.get_sources()),
            'total_consumers': len(self.graph.get_consumers()),
            'flows_with_paths': flows_with_paths,
            'flows_without_paths': flows_without_paths,
        }
    
    def print_all_paths_summary(self, max_per_flow: int = 5):
        """Выводит сводку по всем потокам и их путям."""
        print(f"\n{'='*60}")
        print("СВОДКА ПО ВСЕМ ПОТОКАМ")
        print(f"{'='*60}")
        
        flows_with_paths = [f for f in self.flows if f.paths]
        flows_without_paths = [f for f in self.flows if not f.paths]
        
        print(f"Всего потоков: {len(self.flows)}")
        print(f"  - с найденными путями: {len(flows_with_paths)}")
        print(f"  - без путей: {len(flows_without_paths)}")
        
        if flows_without_paths:
            print(f"\nПотоки без путей:")
            for flow in flows_without_paths:
                print(f"  - {flow.source.name} -> {flow.consumer.name}")
        
        print(f"\n{'='*60}")
        print("ДЕТАЛИ ПО ПОТОКАМ С ПУТЯМИ")
        print(f"{'='*60}")
        
        for flow in flows_with_paths:
            paths_count = len(flow.paths)
            shortest = flow.get_shortest_path()
            
            if shortest is not None:
                min_capacity = self.path_finder.get_path_capacity(shortest)
                shortest_length = len(shortest)
            else:
                min_capacity = 0
                shortest_length = 0
            
            print(f"\n{flow.source.name} -> {flow.consumer.name}:")
            print(f"  Путей: {paths_count}")
            print(f"  Кратчайший путь: {shortest_length} рёбер, мин. способность: {min_capacity:.1f} кВт")
            
            if paths_count <= max_per_flow * 2:
                print(f"  Маршруты:")
                for i, path in enumerate(flow.paths[:max_per_flow], 1):
                    route = self._format_path(path, flow.source)
                    print(f"    {i}. {route}")
                if paths_count > max_per_flow:
                    print(f"    ... и ещё {paths_count - max_per_flow}")
    
    def print_flow_paths(self, flow: Flow, max_display: int = 10):
        """Выводит все пути для заданного потока (для отладки)."""
        print(f"\n{'='*60}")
        print(f"Поток: {flow.source.name} -> {flow.consumer.name}")
        print(f"Найдено путей: {len(flow.paths)}")
        print(f"{'='*60}")
        
        if not flow.paths:
            print("Пути не найдены!")
            return
        
        sorted_paths = sorted(flow.paths, key=len)
        display_count = min(max_display, len(sorted_paths))
        print(f"\nПервые {display_count} путей (отсортированы по длине):\n")
        
        for i, path in enumerate(sorted_paths[:display_count], 1):
            route = self._format_path(path, flow.source)
            length = len(path)
            min_capacity = self.path_finder.get_path_capacity(path)
            
            print(f"{i:2d}. Длина: {length:2d} рёбер, Мин. пропускная способность: {min_capacity:8.1f} кВт")
            print(f"    Маршрут: {route}")
        
        if len(sorted_paths) > max_display:
            print(f"\n... и ещё {len(sorted_paths) - max_display} путей")
    
    def _format_path(self, path: List[Edge], start_node: Node) -> str:
        """Форматирует путь в читаемую строку."""
        if not path:
            return "путь не найден"
        
        route = [start_node.name]
        current = start_node
        
        for edge in path:
            next_node = edge.nodes[0] if edge.nodes[1] == current else edge.nodes[1]
            route.append(next_node.name)
            current = next_node
        
        return " -> ".join(route)