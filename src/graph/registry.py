"""Модуль для управления реестром заявок."""

from typing import List, Dict, Optional
from .model import Graph, Node, Edge
from .request import Request
from .path_finder import PathFinder


class RequestRegistry:
    """Класс для хранения и управления заявками."""
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.requests: List[Request] = []
        self.path_finder = PathFinder(graph)
    
    def generate_all_requests(self) -> int:
        """
        Генерирует заявки для всех возможных пар источник-потребитель.
        """
        sources = self.graph.get_sources()
        consumers = self.graph.get_consumers()
        
        self.requests = []
        for source in sources:
            for consumer in consumers:
                request = Request(source, consumer)
                self.requests.append(request)
        
        return len(self.requests)
    
    def build_all_paths(self, max_depth: int = 50) -> None:
        """
        Строит все возможные простые пути для каждой заявки.
        """
        total_paths = 0
        
        for request in self.requests:
            paths = self.path_finder.find_all_simple_paths(
                request.source, 
                request.consumer, 
                max_depth=max_depth
            )
            request.paths = paths
            total_paths += len(paths)
        
        if self.requests:
            print(f"Построено путей: {total_paths} (в среднем {total_paths/len(self.requests):.1f} на заявку)")
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику по реестру."""
        requests_with_paths = sum(1 for req in self.requests if req.paths)
        requests_without_paths = len(self.requests) - requests_with_paths
        
        return {
            'total_requests': len(self.requests),
            'total_sources': len(self.graph.get_sources()),
            'total_consumers': len(self.graph.get_consumers()),
            'requests_with_paths': requests_with_paths,
            'requests_without_paths': requests_without_paths,
        }
    
    def print_all_paths_summary(self, max_per_request: int = 5):
        """Выводит сводку по всем заявкам и их путям."""
        print(f"\n{'='*60}")
        print("СВОДКА ПО ВСЕМ ЗАЯВКАМ")
        print(f"{'='*60}")
        
        requests_with_paths = [req for req in self.requests if req.paths]
        requests_without_paths = [req for req in self.requests if not req.paths]
        
        print(f"Всего заявок: {len(self.requests)}")
        print(f"  - с найденными путями: {len(requests_with_paths)}")
        print(f"  - без путей: {len(requests_without_paths)}")
        
        if requests_without_paths:
            print(f"\nЗаявки без путей:")
            for req in requests_without_paths:
                print(f"  - {req.source.name} -> {req.consumer.name}")
        
        print(f"\n{'='*60}")
        print("ДЕТАЛИ ПО ЗАЯВКАМ С ПУТЯМИ")
        print(f"{'='*60}")
        
        for req in requests_with_paths:
            paths_count = len(req.paths)
            shortest = req.get_shortest_path()
            
            if shortest is not None:
                min_capacity = self.path_finder.get_path_capacity(shortest)
                shortest_length = len(shortest)
            else:
                min_capacity = 0
                shortest_length = 0
            
            print(f"\n{req.source.name} -> {req.consumer.name}:")
            print(f"  Путей: {paths_count}")
            print(f"  Кратчайший путь: {shortest_length} рёбер, мин. способность: {min_capacity:.1f} кВт")
            
            if paths_count <= max_per_request * 2:
                print(f"  Маршруты:")
                for i, path in enumerate(req.paths[:max_per_request], 1):
                    route = self._format_path(path, req.source)
                    print(f"    {i}. {route}")
                if paths_count > max_per_request:
                    print(f"    ... и ещё {paths_count - max_per_request}")
    
    def print_request_paths(self, request: Request, max_display: int = 10):
        """Выводит все пути для заданной заявки (для отладки)."""
        print(f"\n{'='*60}")
        print(f"Заявка: {request.source.name} -> {request.consumer.name}")
        print(f"Найдено путей: {len(request.paths)}")
        print(f"{'='*60}")
        
        if not request.paths:
            print("Пути не найдены!")
            return
        
        sorted_paths = sorted(request.paths, key=len)
        display_count = min(max_display, len(sorted_paths))
        print(f"\nПервые {display_count} путей (отсортированы по длине):\n")
        
        for i, path in enumerate(sorted_paths[:display_count], 1):
            route = self._format_path(path, request.source)
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