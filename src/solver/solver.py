"""Решатель задачи распределения потоков с помощью градиентного спуска."""

from typing import List, Dict, Tuple, Set
import numpy as np
from graph import Edge, Graph
from .flow_instance import FlowInstance


class Solver:
    """Решает задачу максимизации доставки энергии с учётом пропускных способностей."""
    
    def __init__(self, 
                 graph: Graph, 
                 learning_rate: float = 0.01, 
                 max_iter: int = 1000, 
                 epsilon: float = 1e-6,
                 verbose: bool = True):
        self.graph = graph
        self.lr = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.verbose = verbose
        
        self.instances: List[FlowInstance] = []
        self.loss_history: List[float] = []
        
        # Кэш для быстрого доступа: ребро -> список (instance_idx, path_key, edge_position)
        self._edge_to_flows: Dict[Edge, List[Tuple[int, tuple]]] = {}
    
    def set_instances(self, instances: List[FlowInstance]):
        """Устанавливает список экземпляров потоков."""
        self.instances = instances
        self._build_edge_index()
    
    def _build_edge_index(self):
        """Строит индекс для быстрого определения потоков через каждое ребро."""
        self._edge_to_flows.clear()
        for inst_idx, inst in enumerate(self.instances):
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                for edge in path:
                    if edge not in self._edge_to_flows:
                        self._edge_to_flows[edge] = []
                    self._edge_to_flows[edge].append((inst_idx, path_key))
    
    def initialize_uniform(self):
        """Инициализирует равномерное распределение целевого потока по путям."""
        for inst in self.instances:
            inst.set_uniform_flow()
    
    def compute_edge_total_flow(self, edge: Edge) -> float:
        """Вычисляет суммарный поток через данное ребро."""
        total = 0.0
        if edge in self._edge_to_flows:
            for inst_idx, path_key in self._edge_to_flows[edge]:
                inst = self.instances[inst_idx]
                total += inst.path_flows.get(path_key, 0.0)
        return total
    
    def compute_loss(self) -> Tuple[float, Dict[str, float]]:
        """
        Вычисляет значение функции потерь.
        Возвращает (total_loss, components) где components содержит слагаемые.
        """
        loss_capacity = 0.0
        loss_demand = 0.0
        
        # Штраф за превышение пропускной способности
        for edge in self.graph.edges:
            if edge.capacity == float('inf'):
                continue
            flow = self.compute_edge_total_flow(edge)
            if flow > edge.capacity:
                excess_ratio = flow - edge.capacity
                loss_capacity += excess_ratio ** 2
        
        # Штраф за недопоставку энергии
        for inst in self.instances:
            total_flow = inst.get_total_flow()
            if inst.target_amount > 0:
                shortage_ratio = total_flow - inst.target_amount
                loss_demand += shortage_ratio ** 2
        
        total_loss = loss_capacity + loss_demand
        components = {
            'capacity_violation': loss_capacity,
            'demand_shortage': loss_demand
        }
        return total_loss, components
    
    def compute_gradients(self) -> Dict[Tuple[int, tuple], float]:
        """
        Вычисляет градиенты функции потерь по каждой переменной path_flow.
        Возвращает словарь: (inst_idx, path_key) -> градиент.
        """
        gradients: Dict[Tuple[int, tuple], float] = {}
        
        # Инициализируем нулями
        for inst_idx, inst in enumerate(self.instances):
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                gradients[(inst_idx, path_key)] = 0.0
        
        # Градиент от штрафа за пропускную способность
        for edge in self.graph.edges:
            if edge.capacity == float('inf'):
                continue
            flow = self.compute_edge_total_flow(edge)
            if flow <= edge.capacity:
                continue  # градиент = 0, если не превышена
            
            # Производная: d/dx_i [ (flow - cap)^2 ] = 2 * (flow - cap)
            grad_factor = 2.0 * (flow - edge.capacity)
            
            if edge in self._edge_to_flows:
                for inst_idx, path_key in self._edge_to_flows[edge]:
                    gradients[(inst_idx, path_key)] += grad_factor
        
        # Градиент от штрафа за недопоставку
        for inst_idx, inst in enumerate(self.instances):
            total_flow = inst.get_total_flow()
            if inst.target_amount <= 0:
                continue
            shortage = total_flow - inst.target_amount
            
            # Производная: d/dx_i [ (sum - target)^2 ] = 2 * (sum - target)
            grad_factor = 2.0 * shortage
            
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                gradients[(inst_idx, path_key)] += grad_factor
        
        return gradients
    
    def optimize(self) -> Dict:
        """Выполняет градиентный спуск."""
        if not self.instances:
            return {'success': False, 'message': 'Нет экземпляров потоков'}
        
        self.initialize_uniform()
        self.loss_history = []
        
        for iteration in range(self.max_iter):
            # Вычисляем текущие потери и градиенты
            loss, components = self.compute_loss()
            self.loss_history.append(loss)
            
            if self.verbose and iteration % 100 == 0:
                print(f"Итерация {iteration:4d}: loss = {loss:.6f} "
                      f"(cap={components['capacity_violation']:.6f}, "
                      f"demand={components['demand_shortage']:.6f})")
            
            # Проверка сходимости
            if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.epsilon:
                if self.verbose:
                    print(f"Сошлось на итерации {iteration} с loss = {loss:.6f}")
                break
            
            # Вычисляем градиенты
            grads = self.compute_gradients()
            
            # Обновляем потоки (градиентный спуск)
            for (inst_idx, path_key), grad in grads.items():
                if abs(grad) > 1e-12:
                    inst = self.instances[inst_idx]
                    # Находим путь по ключу (обратное отображение)
                    for path in inst.get_paths():
                        if inst._path_to_key(path) == path_key:
                            delta = -self.lr * grad
                            inst.update_path_flow(path, delta)
                            break
        
        final_loss, final_components = self.compute_loss()
        return {
            'success': True,
            'final_loss': final_loss,
            'capacity_violation': final_components['capacity_violation'],
            'demand_shortage': final_components['demand_shortage'],
            'iterations': len(self.loss_history),
            'loss_history': self.loss_history
        }
    
    def get_edge_loads(self) -> Dict[Edge, float]:
        """Возвращает словарь с загрузкой каждого ребра (поток / capacity)."""
        loads = {}
        for edge in self.graph.edges:
            flow = self.compute_edge_total_flow(edge)
            if edge.capacity == float('inf'):
                loads[edge] = 0.0
            else:
                loads[edge] = flow / edge.capacity if edge.capacity > 0 else 0.0
        return loads
    
    def get_delivery_report(self) -> Dict:
        """Возвращает отчёт о доставке энергии по заявкам."""
        report = []
        total_delivered = 0.0
        total_requested = 0.0
        
        for inst in self.instances:
            delivered = inst.get_total_flow()
            requested = inst.target_amount
            shortage = requested - delivered
            shortage_pct = (shortage / requested * 100) if requested > 0 else 0.0
            
            report.append({
                'source': inst.request.source.name,
                'consumer': inst.request.consumer.name,
                'requested': requested,
                'delivered': delivered,
                'shortage': shortage,
                'shortage_pct': shortage_pct
            })
            total_delivered += delivered
            total_requested += requested
        
        return {
            'items': report,
            'total_requested': total_requested,
            'total_delivered': total_delivered,
            'total_shortage': total_requested - total_delivered,
            'delivery_ratio': total_delivered / total_requested if total_requested > 0 else 0.0
        }
    
    def get_edge_violations(self) -> List[Dict]:
        """Возвращает список рёбер с превышением пропускной способности."""
        violations = []
        for edge in self.graph.edges:
            if edge.capacity == float('inf'):
                continue
            flow = self.compute_edge_total_flow(edge)
            if flow > edge.capacity:
                violations.append({
                    'edge': f"{edge.nodes[0].name}-{edge.nodes[1].name}",
                    'capacity': edge.capacity,
                    'actual_flow': flow,
                    'excess': flow - edge.capacity,
                    'excess_pct': (flow - edge.capacity) / edge.capacity * 100
                })
        return violations