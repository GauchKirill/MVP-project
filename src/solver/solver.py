"""Решатель задачи распределения потоков с помощью градиентного спуска."""

from typing import List, Dict, Tuple
from graph import Edge, Graph
from .flow_instance import FlowInstance


class Solver:
    """Решает задачу максимизации доставки энергии с учётом пропускных способностей."""
    
    def __init__(self, 
                graph: Graph, 
                learning_rate: float = 0.01, 
                max_iter: int = 1000, 
                epsilon: float = 1e-6,
                gradient_epsilon_rel: float = 0.01,  # 1% от текущего значения
                verbose: bool = True):
        self.graph = graph
        self.lr = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.gradient_epsilon_rel = gradient_epsilon_rel
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
    
    def compute_actual_flows(self, desired_flows: Dict[Tuple[int, tuple], float]) -> Dict[Tuple[int, tuple], float]:
        """
        Вычисляет фактические потоки после применения ограничений пропускной способности
        и правила пропорционального ограничения.
        """
        actual_flows = {}
        for key, flow in desired_flows.items():
            actual_flows[key] = flow
        
        # Шаг 1: Применяем ограничения пропускной способности рёбер
        edge_flows: Dict[Edge, float] = {}
        for edge in self.graph.edges:
            edge_flows[edge] = 0.0
        
        for (inst_idx, path_key), flow in actual_flows.items():
            inst = self.instances[inst_idx]
            for path in inst.get_paths():
                if inst._path_to_key(path) == path_key:
                    for edge in path:
                        edge_flows[edge] = edge_flows.get(edge, 0.0) + flow
                    break
        
        # Определяем коэффициент масштабирования для рёбер с превышением
        edge_scale_factors: Dict[Edge, float] = {}
        for edge, flow in edge_flows.items():
            if edge.capacity != float('inf') and flow > edge.capacity:
                edge_scale_factors[edge] = edge.capacity / flow
            else:
                edge_scale_factors[edge] = 1.0
        
        # Применяем масштабирование к путям
        for (inst_idx, path_key), flow in actual_flows.items():
            inst = self.instances[inst_idx]
            for path in inst.get_paths():
                if inst._path_to_key(path) == path_key:
                    min_scale = 1.0
                    for edge in path:
                        if edge in edge_scale_factors:
                            min_scale = min(min_scale, edge_scale_factors[edge])
                    actual_flows[(inst_idx, path_key)] = flow * min_scale
                    break
        
        # Шаг 2: Применяем правило пропорционального ограничения для потребителей
        consumer_actual: Dict[str, Dict[int, float]] = {}   # consumer -> {inst_idx: actual_flow}
        consumer_targets: Dict[str, Dict[int, float]] = {}  # consumer -> {inst_idx: target_amount}
        
        for inst_idx, inst in enumerate(self.instances):
            consumer_name = inst.request.consumer.name
            if consumer_name not in consumer_actual:
                consumer_actual[consumer_name] = {}
                consumer_targets[consumer_name] = {}
            
            # Суммируем фактические потоки по всем путям
            actual_total = 0.0
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                actual_total += actual_flows.get((inst_idx, path_key), 0.0)
            
            consumer_actual[consumer_name][inst_idx] = actual_total
            consumer_targets[consumer_name][inst_idx] = inst.target_amount
        
        # Для каждого потребителя проверяем правило пропорционального ограничения
        for consumer_name in consumer_actual:
            total_actual = sum(consumer_actual[consumer_name].values())
            total_target = sum(consumer_targets[consumer_name].values())
            
            # Проверяем, есть ли ограничение (если total_actual < сумма желаемых)
            # Сравниваем с желаемыми потоками до ограничений
            total_desired = 0.0
            for inst_idx in consumer_actual[consumer_name]:
                inst = self.instances[inst_idx]
                for path in inst.get_paths():
                    path_key = inst._path_to_key(path)
                    total_desired += desired_flows.get((inst_idx, path_key), 0.0)
            
            if total_actual < total_desired and total_target > 0:
                # Перераспределяем total_actual пропорционально заявленным мощностям
                for inst_idx in consumer_actual[consumer_name]:
                    inst = self.instances[inst_idx]
                    ideal_share = inst.target_amount / total_target
                    proportional_flow = ideal_share * total_actual
                    
                    current_actual = consumer_actual[consumer_name][inst_idx]
                    if current_actual > 0:
                        scale = proportional_flow / current_actual
                    else:
                        scale = 0.0
                    
                    for path in inst.get_paths():
                        path_key = inst._path_to_key(path)
                        if (inst_idx, path_key) in actual_flows:
                            actual_flows[(inst_idx, path_key)] *= scale
        
        return actual_flows

    def compute_loss(self) -> Tuple[float, Dict[str, float]]:
        """
        Вычисляет значение функции потерь.
        Loss = сумма по всем заявкам (target_amount - actual_flow)
        (только положительная недопоставка)
        """
        # Собираем желаемые потоки из текущего состояния
        desired_flows: Dict[Tuple[int, tuple], float] = {}
        for inst_idx, inst in enumerate(self.instances):
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                desired_flows[(inst_idx, path_key)] = inst.path_flows[path_key]
        
        # Вычисляем фактические потоки после всех ограничений
        actual_flows = self.compute_actual_flows(desired_flows)
        
        # Считаем потери
        total_loss = 0.0
        loss_details = {}
        
        for inst_idx, inst in enumerate(self.instances):
            # Суммарный фактический поток для данной заявки
            actual_total = 0.0
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                actual_total += actual_flows.get((inst_idx, path_key), 0.0)
            
            # Недопоставка = заявленная - фактическая
            shortage = abs(inst.target_amount - actual_total)
            total_loss += shortage
            
            loss_details[f"{inst.request.source.name}->{inst.request.consumer.name}"] = {
                'target': inst.target_amount,
                'actual': actual_total,
                'shortage': shortage
            }
        
        components = {
            'total_loss': total_loss,
            'details': loss_details
        }
        return total_loss, components
    
    def compute_gradients(self) -> Dict[Tuple[int, tuple], float]:
        """
        Вычисляет градиенты функции потерь.
        Использует конечные разности с относительным шагом.
        """
        gradients: Dict[Tuple[int, tuple], float] = {}
        
        # Инициализируем нулями
        for inst_idx, inst in enumerate(self.instances):
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                gradients[(inst_idx, path_key)] = 0.0
        
        # Сохраняем текущее состояние
        current_state = {}
        for inst_idx, inst in enumerate(self.instances):
            current_state[inst_idx] = inst.path_flows.copy()
        
        # Вычисляем текущие потери
        current_loss, _ = self.compute_loss()
        
        for inst_idx, inst in enumerate(self.instances):
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                current_value = inst.path_flows[path_key]
                
                # Относительный шаг, но не менее 0.1 кВт
                epsilon = max(current_value * self.gradient_epsilon_rel, 0.1)
                
                if current_value < 0.1:
                    continue
                
                # Возмущаем вверх
                inst.path_flows[path_key] = current_value + epsilon
                loss_up, _ = self.compute_loss()
                
                # Восстанавливаем
                inst.path_flows[path_key] = current_value
                
                # Градиент
                grad = (loss_up - current_loss) / epsilon
                gradients[(inst_idx, path_key)] = grad
        
        # Восстанавливаем состояние
        for inst_idx, inst in enumerate(self.instances):
            inst.path_flows = current_state[inst_idx]
        
        return gradients
    
    def optimize(self) -> Dict:
        """Выполняет градиентный спуск."""
        if not self.instances:
            return {'success': False, 'message': 'Нет экземпляров потоков'}
        
        self.initialize_uniform()
        self.loss_history = []
        
        for iteration in range(self.max_iter):
            # Вычисляем текущие потери
            loss, components = self.compute_loss()
            self.loss_history.append(loss)
            
            if self.verbose and iteration % 5 == 0:
                # Получаем детали для вывода
                desired_flows = {}
                for inst_idx, inst in enumerate(self.instances):
                    for path in inst.get_paths():
                        path_key = inst._path_to_key(path)
                        desired_flows[(inst_idx, path_key)] = inst.path_flows[path_key]
                actual_flows = self.compute_actual_flows(desired_flows)
                
                # Считаем превышения пропускной способности
                edge_flows = {}
                for edge in self.graph.edges:
                    edge_flows[edge] = 0.0
                for (inst_idx, path_key), flow in actual_flows.items():
                    inst = self.instances[inst_idx]
                    for path in inst.get_paths():
                        if inst._path_to_key(path) == path_key:
                            for edge in path:
                                edge_flows[edge] = edge_flows.get(edge, 0.0) + flow
                            break
                
                capacity_violations = 0.0
                for edge, flow in edge_flows.items():
                    if edge.capacity != float('inf') and flow > edge.capacity:
                        capacity_violations += flow - edge.capacity
                
                # Считаем суммарную недопоставку
                total_shortage = 0.0
                for inst_idx, inst in enumerate(self.instances):
                    actual_total = sum(actual_flows.get((inst_idx, inst._path_to_key(p)), 0.0) 
                                    for p in inst.get_paths())
                    if actual_total < inst.target_amount:
                        total_shortage += inst.target_amount - actual_total
                
                print(f"Итерация {iteration:4d}: loss = {loss:.2f} кВт "
                    f"(недопоставка={total_shortage:.2f}, превышений={capacity_violations:.2f})")
            
            # Проверка сходимости
            if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.epsilon:
                if self.verbose:
                    print(f"Сошлось на итерации {iteration} с loss = {loss:.2f} кВт")
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
        
        # Финальные метрики
        final_loss, final_components = self.compute_loss()
        
        # Считаем финальные метрики
        desired_flows = {}
        for inst_idx, inst in enumerate(self.instances):
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                desired_flows[(inst_idx, path_key)] = inst.path_flows[path_key]
        actual_flows = self.compute_actual_flows(desired_flows)
        
        total_shortage = 0.0
        for inst_idx, inst in enumerate(self.instances):
            actual_total = sum(actual_flows.get((inst_idx, inst._path_to_key(p)), 0.0) 
                            for p in inst.get_paths())
            if actual_total < inst.target_amount:
                total_shortage += inst.target_amount - actual_total
        
        edge_flows = {}
        for edge in self.graph.edges:
            edge_flows[edge] = 0.0
        for (inst_idx, path_key), flow in actual_flows.items():
            inst = self.instances[inst_idx]
            for path in inst.get_paths():
                if inst._path_to_key(path) == path_key:
                    for edge in path:
                        edge_flows[edge] = edge_flows.get(edge, 0.0) + flow
                    break
        
        capacity_violations = 0.0
        for edge, flow in edge_flows.items():
            if edge.capacity != float('inf') and flow > edge.capacity:
                capacity_violations += flow - edge.capacity
        
        return {
            'success': True,
            'final_loss': final_loss,
            'total_shortage': total_shortage,
            'capacity_violation': capacity_violations,
            'iterations': len(self.loss_history),
            'loss_history': self.loss_history
        }
    
    def get_edge_loads(self) -> Dict[Edge, Tuple[float, float]]:
        """
        Возвращает словарь с фактическим потоком и загрузкой каждого ребра.
        Returns: {edge: (flow, load_ratio)}
        """
        loads = {}
        
        # Получаем фактические потоки
        desired_flows = {}
        for inst_idx, inst in enumerate(self.instances):
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                desired_flows[(inst_idx, path_key)] = inst.path_flows[path_key]
        actual_flows = self.compute_actual_flows(desired_flows)
        
        # Суммируем потоки по рёбрам
        edge_flows = {}
        for edge in self.graph.edges:
            edge_flows[edge] = 0.0
        
        for (inst_idx, path_key), flow in actual_flows.items():
            inst = self.instances[inst_idx]
            for path in inst.get_paths():
                if inst._path_to_key(path) == path_key:
                    for edge in path:
                        edge_flows[edge] = edge_flows.get(edge, 0.0) + flow
                    break
        
        for edge in self.graph.edges:
            flow = edge_flows.get(edge, 0.0)
            if edge.capacity == float('inf'):
                loads[edge] = (flow, 0.0)
            else:
                load_ratio = flow / edge.capacity if edge.capacity > 0 else 0.0
                loads[edge] = (flow, load_ratio)
        
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
    
    def plot_training_history(self, filename="training_history.png"):
        """Строит графики процесса обучения."""
        import matplotlib.pyplot as plt
        
        if not self.loss_history:
            print("Нет истории обучения")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Loss от итерации
        ax1 = axes[0, 0]
        ax1.plot(self.loss_history, 'b-', linewidth=1)
        ax1.set_xlabel('Итерация')
        ax1.set_ylabel('Loss (кВт)')
        ax1.set_title('Функция потерь')
        ax1.grid(True, alpha=0.3)
        
        # 2. Loss в логарифмическом масштабе
        ax2 = axes[0, 1]
        ax2.semilogy(self.loss_history, 'r-', linewidth=1)
        ax2.set_xlabel('Итерация')
        ax2.set_ylabel('Loss (кВт, log scale)')
        ax2.set_title('Функция потерь (логарифмический масштаб)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Изменение loss (градиент)
        ax3 = axes[1, 0]
        if len(self.loss_history) > 1:
            loss_diff = [abs(self.loss_history[i] - self.loss_history[i-1]) 
                        for i in range(1, len(self.loss_history))]
            ax3.plot(loss_diff, 'g-', linewidth=0.5)
            ax3.set_xlabel('Итерация')
            ax3.set_ylabel('|ΔLoss| (кВт)')
            ax3.set_title('Изменение функции потерь')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=self.epsilon, color='r', linestyle='--', label=f'ε = {self.epsilon}')
            ax3.legend()
        
        # 4. Относительное изменение loss
        ax4 = axes[1, 1]
        if len(self.loss_history) > 1 and self.loss_history[0] > 0:
            relative_change = [abs(self.loss_history[i] - self.loss_history[i-1]) / self.loss_history[i-1] 
                            for i in range(1, len(self.loss_history))]
            ax4.semilogy(relative_change, 'm-', linewidth=0.5)
            ax4.set_xlabel('Итерация')
            ax4.set_ylabel('|ΔLoss| / Loss')
            ax4.set_title('Относительное изменение функции потерь')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.show()
        print(f"Графики обучения сохранены в {filename}")