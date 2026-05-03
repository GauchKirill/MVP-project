import copy
from typing import List, Dict, Tuple
from graph import Edge, Graph
import matplotlib.pyplot as plt

from .flow_instance import FlowInstance

class Solver:
    """Решает задачу максимизации доставки энергии с учётом пропускных способностей."""

    def __init__(self,
                 graph: Graph,
                 learning_rate: float = 0.01,
                 max_iter: int = 1000,
                 epsilon: float = 1e-6,
                 gradient_epsilon_rel: float = 0.01,
                 capacity_weight: float = 10.0,
                 early_stopping_patience: int = 20,
                 verbose: bool = True):
        self.graph = graph
        self.lr = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.gradient_epsilon_rel = gradient_epsilon_rel
        self.capacity_weight = capacity_weight
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose

        self.instances: List[FlowInstance] = []
        self.loss_history: List[float] = []

        # Кэш: ребро -> список (inst_idx, path_key)
        self._edge_to_flows: Dict[Edge, List[Tuple[int, tuple]]] = {}

    def set_instances(self, instances: List[FlowInstance]):
        """Устанавливает список экземпляров потоков"""
        self.instances = instances
        self._build_edge_index()

    def _build_edge_index(self):
        """Строит индекс для быстрого определения потоков через каждое ребро"""
        self._edge_to_flows.clear()
        for inst_idx, inst in enumerate(self.instances):
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                for edge in path:
                    if edge not in self._edge_to_flows:
                        self._edge_to_flows[edge] = []
                    self._edge_to_flows[edge].append((inst_idx, path_key))

    def initialize_uniform(self):
        """Инициализирует равномерное распределение целевого потока по путям"""
        for inst in self.instances:
            inst.set_uniform_flow()
    
    def compute_edge_total_flow(self, edge: Edge) -> float:
        """Вычисляет суммарный поток через данное ребро"""
        total = 0.0
        if edge in self._edge_to_flows:
            for inst_idx, path_key in self._edge_to_flows[edge]:
                inst = self.instances[inst_idx]
                total += inst.path_flows.get(path_key, 0.0)
        return total
    
    def compute_actual_flows(self, desired_flows: Dict[Tuple[int, tuple], float]) -> Dict[Tuple[int, tuple], float]:
        """
        Вычисляет реальные потоки после последовательного применения ограничений:
        1. Пропускная способность рёбер (итеративно, до схождения).
        2. Ограничение сверху: приход к потребителю ≤ сумма заявок на него.
        3. Пропорциональное распределение при недопоставке.
        """
        actual = copy.deepcopy(desired_flows)

        # ---------- 1. Итеративное ограничение по пропускной способности рёбер ----------
        max_iter_cap = 100
        for _ in range(max_iter_cap):
            # Вычисляем суммарный поток на каждом ребре
            edge_loads = {edge: 0.0 for edge in self.graph.edges}
            for (inst_idx, path_key), flow in actual.items():
                inst = self.instances[inst_idx]
                for path in inst.get_paths():
                    if inst._path_to_key(path) == path_key:
                        for edge in path:
                            edge_loads[edge] += flow
                        break

            # Собираем коэффициенты масштабирования для перегруженных рёбер
            scale_per_edge = {}
            for edge, total_flow in edge_loads.items():
                if edge.capacity != float('inf') and total_flow > edge.capacity:
                    scale_per_edge[edge] = edge.capacity / total_flow
                else:
                    scale_per_edge[edge] = 1.0

            # Если ни одно конечное ребро не перегружено – выходим
            all_clear = True
            for edge, s in scale_per_edge.items():
                if edge.capacity != float('inf') and abs(s - 1.0) >= 1e-9:
                    all_clear = False
                    break
            if all_clear:
                break

            # Для каждого пути находим минимальный коэффициент по всем его рёбрам
            new_actual = {}
            for (inst_idx, path_key), flow in actual.items():
                inst = self.instances[inst_idx]
                for path in inst.get_paths():
                    if inst._path_to_key(path) == path_key:
                        min_scale = min(scale_per_edge[edge] for edge in path)
                        new_actual[(inst_idx, path_key)] = flow * min_scale
                        break
            actual = new_actual

        # ---------- 2. Ограничение сверху по потребителям ----------
        # Собираем целевые суммы по потребителям
        consumer_targets = {}
        for inst_idx, inst in enumerate(self.instances):
            c_name = inst.request.consumer.name
            consumer_targets[c_name] = consumer_targets.get(c_name, 0.0) + inst.target_amount

        # Считаем фактический приход к каждому потребителю
        consumer_actual = {}
        for (inst_idx, path_key), flow in actual.items():
            inst = self.instances[inst_idx]
            c_name = inst.request.consumer.name
            consumer_actual[c_name] = consumer_actual.get(c_name, 0.0) + flow

        # Если приход больше, чем сумма заявок – пропорционально урезаем
        for c_name, actual_sum in consumer_actual.items():
            target_sum = consumer_targets[c_name]
            if actual_sum > target_sum and actual_sum > 0:
                scale = target_sum / actual_sum
                for (inst_idx, path_key), flow in actual.items():
                    inst = self.instances[inst_idx]
                    if inst.request.consumer.name == c_name:
                        actual[(inst_idx, path_key)] = flow * scale

        # ---------- 3. Пропорциональное распределение при недопоставке ----------
        # После ограничения сверху снова пересчитываем фактический приход
        consumer_actual = {}
        for (inst_idx, path_key), flow in actual.items():
            inst = self.instances[inst_idx]
            c_name = inst.request.consumer.name
            consumer_actual[c_name] = consumer_actual.get(c_name, 0.0) + flow

        for c_name, actual_sum in consumer_actual.items():
            target_sum = consumer_targets[c_name]
            if actual_sum < target_sum and target_sum > 0:
                # Пропорционально заявкам распределяем actual_sum
                for (inst_idx, path_key), flow in actual.items():
                    inst = self.instances[inst_idx]
                    if inst.request.consumer.name == c_name:
                        ideal = (inst.target_amount / target_sum) * actual_sum
                        # Масштабируем текущий путь пропорционально
                        # Находим все пути этой заявки
                        inst_paths = []
                        for p in inst.get_paths():
                            pk = inst._path_to_key(p)
                            if (inst_idx, pk) in actual:
                                inst_paths.append(pk)
                        total_inst_flow = sum(actual.get((inst_idx, pk), 0.0) for pk in inst_paths)
                        if total_inst_flow > 0:
                            scale = ideal / total_inst_flow
                            for pk in inst_paths:
                                actual[(inst_idx, pk)] *= scale
                        # Если потоков не было, но ideal > 0 – не можем создать из ничего, оставляем 0.

        return actual

    def compute_loss(self) -> Tuple[float, Dict[str, float]]:
        """Лосс: недопоставка + превышение capacity."""
        desired_flows = {}
        for inst_idx, inst in enumerate(self.instances):
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                desired_flows[(inst_idx, path_key)] = inst.path_flows[path_key]

        actual_flows = self.compute_actual_flows(desired_flows)

        total_loss = 0.0
        components = {}

        # 1. Недопоставка
        shortage = 0.0
        for inst_idx, inst in enumerate(self.instances):
            inst_paths = [inst._path_to_key(p) for p in inst.get_paths()]
            actual_total = sum(actual_flows.get((inst_idx, pk), 0.0) for pk in inst_paths)
            target = inst.target_amount
            if actual_total < target:
                shortage += target - actual_total
        total_loss += shortage
        components['shortage'] = shortage

        # 2. Превышение capacity
        edge_flows = {edge: 0.0 for edge in self.graph.edges}
        for (inst_idx, path_key), flow in actual_flows.items():
            inst = self.instances[inst_idx]
            for path in inst.get_paths():
                if inst._path_to_key(path) == path_key:
                    for edge in path:
                        edge_flows[edge] += flow
                    break

        capacity_violation = 0.0
        for edge, flow in edge_flows.items():
            if edge.capacity != float('inf') and flow > edge.capacity:
                capacity_violation += flow - edge.capacity
        total_loss += self.capacity_weight * capacity_violation
        components['capacity_violation'] = capacity_violation

        components['total_loss'] = total_loss
        return total_loss, components

    def compute_gradients(self) -> Dict[Tuple[int, tuple], float]:
        """Градиенты конечными разностями."""
        gradients = {}
        for inst_idx, inst in enumerate(self.instances):
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                gradients[(inst_idx, path_key)] = 0.0

        # Сохраняем текущее состояние
        current_state = {}
        for inst_idx, inst in enumerate(self.instances):
            current_state[inst_idx] = inst.path_flows.copy()

        current_loss, _ = self.compute_loss()

        for inst_idx, inst in enumerate(self.instances):
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                current_value = inst.path_flows[path_key]

                eps = max(current_value * self.gradient_epsilon_rel, 0.1)
                if current_value < 0.1:
                    continue

                # Возмущение вверх
                inst.path_flows[path_key] = current_value + eps
                loss_up, _ = self.compute_loss()

                # Возвращаем
                inst.path_flows[path_key] = current_value

                gradients[(inst_idx, path_key)] = (loss_up - current_loss) / eps

        # Восстанавливаем состояние
        for inst_idx, inst in enumerate(self.instances):
            inst.path_flows = current_state[inst_idx]

        return gradients

    def _apply_two_stage_update(self, gradients: Dict[Tuple[int, tuple], float]):
        """
        Двухэтапное обновление на каждой итерации:
        Этап 1 – корректировка суммарного потока по заявке.
        Этап 2 – перераспределение между путями без изменения суммарного потока.
        """
        # --- Этап 1: изменение total_flow каждой заявки ---
        # Вычисляем градиент по total_flow: сумма градиентов по всем путям заявки
        total_grads = {}
        for inst_idx, inst in enumerate(self.instances):
            inst_grad = 0.0
            for path in inst.get_paths():
                key = (inst_idx, inst._path_to_key(path))
                inst_grad += gradients.get(key, 0.0)
            total_grads[inst_idx] = inst_grad

        # Применяем обновление к total_flow, затем пропорционально масштабируем пути
        for inst_idx, inst in enumerate(self.instances):
            current_total = inst.get_total_flow()
            grad = total_grads[inst_idx]
            new_total = current_total - self.lr * grad
            # Ограничиваем диапазон [0, target_amount]
            new_total = max(0.0, min(new_total, inst.target_amount))
            if current_total > 0:
                scale = new_total / current_total
            else:
                # Если не было потока, но градиент отрицательный – можно инициализировать равномерно
                if new_total > 0:
                    inst.set_uniform_flow()
                    # Масштабируем до new_total
                    scale = new_total / inst.get_total_flow()
                else:
                    continue
            # Масштабируем все пути
            for path in inst.get_paths():
                key = inst._path_to_key(path)
                inst.path_flows[key] *= scale

        # --- Этап 2: перераспределение между путями ---
        # Вычисляем средний градиент по заявке и обновляем пути, убрав среднее
        for inst_idx, inst in enumerate(self.instances):
            paths = inst.get_paths()
            if len(paths) <= 1:
                continue
            grad_values = []
            for path in paths:
                key = (inst_idx, inst._path_to_key(path))
                grad_values.append(gradients.get(key, 0.0))
            avg_grad = sum(grad_values) / len(grad_values)

            # Обновляем каждый путь с «центрированным» градиентом
            for path, g in zip(paths, grad_values):
                centered_grad = g - avg_grad
                key = (inst_idx, inst._path_to_key(path))
                delta = -self.lr * centered_grad
                inst.update_path_flow(path, delta)

            # Корректируем сумму, чтобы осталась равной total (на случай погрешностей)
            target_total = inst.get_total_flow()
            current_total = sum(inst.path_flows.values())
            if current_total > 0 and abs(current_total - target_total) > 1e-9:
                scale = target_total / current_total
                for path in paths:
                    key = inst._path_to_key(path)
                    inst.path_flows[key] *= scale

    def optimize(self) -> Dict:
        """Выполняет градиентный спуск"""
        if not self.instances:
            return {'success': False, 'message': 'Нет экземпляров потоков'}

        self.loss_history = []
        
        loss = 0.0
        components = {'shortage': 0.0, 'capacity_violation': 0.0, 'total_loss': 0.0}

        # Для ранней остановки: отслеживаем лучшее состояние
        best_loss = float('inf')
        best_state = None
        best_iteration = 0

        for iteration in range(self.max_iter):
            loss, components = self.compute_loss()
            self.loss_history.append(loss)

            # Сохраняем лучшее состояние
            if loss < best_loss:
                best_loss = loss
                best_iteration = iteration
                # Глубокая копия текущих потоков
                best_state = {}
                for inst_idx, inst in enumerate(self.instances):
                    best_state[inst_idx] = inst.path_flows.copy()

            if self.verbose and iteration % 5 == 0:
                print(f"Итерация {iteration:4d}: loss = {loss:.2f} "
                      f"(shortage={components['shortage']:.2f}, "
                      f"cap_viol={components['capacity_violation']:.2f})")

            # Проверка сходимости
            if iteration > 0:
                # Критерий 1: абсолютное изменение меньше epsilon
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.epsilon:
                    if self.verbose:
                        print(f"Сошлось по epsilon на итерации {iteration} с loss = {loss:.2f}")
                    break
                
                # Критерий 2: нет улучшения на протяжении early_stopping_patience итераций
                if iteration >= self.early_stopping_patience:
                    # Проверяем, когда был достигнут лучший результат
                    if iteration - best_iteration >= self.early_stopping_patience:
                        if self.verbose:
                            print(f"Ранняя остановка на итерации {iteration}: "
                                  f"нет улучшения {self.early_stopping_patience} итераций "
                                  f"(лучший loss = {best_loss:.2f} на итерации {best_iteration})")
                        break

            grads = self.compute_gradients()
            self._apply_two_stage_update(grads)

        # Восстанавливаем лучшее состояние
        if best_state is not None:
            for inst_idx, inst in enumerate(self.instances):
                inst.path_flows = best_state[inst_idx]

        # Финальные метрики — считаем на лучшем состоянии
        desired_flows = {}
        for inst_idx, inst in enumerate(self.instances):
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                desired_flows[(inst_idx, path_key)] = inst.path_flows[path_key]

        actual_flows = self.compute_actual_flows(desired_flows)

        total_shortage = 0.0
        total_excess = 0.0
        for inst_idx, inst in enumerate(self.instances):
            inst_paths = [inst._path_to_key(p) for p in inst.get_paths()]
            actual_total = sum(actual_flows.get((inst_idx, pk), 0.0) for pk in inst_paths)
            if actual_total < inst.target_amount:
                total_shortage += inst.target_amount - actual_total
            elif actual_total > inst.target_amount:
                total_excess += actual_total - inst.target_amount

        capacity_violation = 0.0
        edge_flows = {edge: 0.0 for edge in self.graph.edges}
        for (inst_idx, path_key), flow in actual_flows.items():
            inst = self.instances[inst_idx]
            for path in inst.get_paths():
                if inst._path_to_key(path) == path_key:
                    for edge in path:
                        edge_flows[edge] += flow
                    break
        for edge, flow in edge_flows.items():
            if edge.capacity != float('inf') and flow > edge.capacity:
                capacity_violation += flow - edge.capacity

        return {
            'success': True,
            'final_loss': best_loss,
            'total_shortage': total_shortage,
            'total_excess': total_excess,
            'capacity_violation': capacity_violation,
            'iterations': len(self.loss_history),
            'best_iteration': best_iteration,
            'loss_history': self.loss_history
        }

    def get_edge_loads(self) -> Dict[Edge, Tuple[float, float]]:
        """
        Возвращает словарь с фактическим потоком и загрузкой каждого ребра
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

        edge_flows = {edge: 0.0 for edge in self.graph.edges}
        for (inst_idx, path_key), flow in actual_flows.items():
            inst = self.instances[inst_idx]
            for path in inst.get_paths():
                if inst._path_to_key(path) == path_key:
                    for edge in path:
                        edge_flows[edge] += flow
                    break

        loads = {}
        for edge in self.graph.edges:
            flow = edge_flows[edge]
            if edge.capacity == float('inf'):
                loads[edge] = (flow, 0.0)
            else:
                ratio = flow / edge.capacity if edge.capacity > 0 else 0.0
                loads[edge] = (flow, ratio)
        return loads

    def get_delivery_report(self) -> Dict:
        """Отчёт о доставке (на основе фактических потоков)."""
        desired_flows = {}
        for inst_idx, inst in enumerate(self.instances):
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                desired_flows[(inst_idx, path_key)] = inst.path_flows[path_key]
        actual_flows = self.compute_actual_flows(desired_flows)

        report = []
        total_delivered = 0.0
        total_requested = 0.0
        for inst_idx, inst in enumerate(self.instances):
            inst_paths = [inst._path_to_key(p) for p in inst.get_paths()]
            delivered = sum(actual_flows.get((inst_idx, pk), 0.0) for pk in inst_paths)
            requested = inst.target_amount
            shortage = max(requested - delivered, 0.0)
            excess = max(delivered - requested, 0.0)
            report.append({
                'source': inst.request.source.name,
                'consumer': inst.request.consumer.name,
                'requested': requested,
                'delivered': delivered,
                'shortage': shortage,
                'excess': excess
            })
            total_delivered += delivered
            total_requested += requested

        return {
            'items': report,
            'total_requested': total_requested,
            'total_delivered': total_delivered,
            'total_shortage': total_requested - total_delivered if total_delivered <= total_requested else 0.0,
            'total_excess': max(total_delivered - total_requested, 0.0),
            'delivery_ratio': total_delivered / total_requested if total_requested > 0 else 0.0
        }
    
    def get_desired_directed_flows(self) -> Dict[Tuple[str, str], float]:
        """
        Возвращает желаемые направленные потоки (до применения ограничений)
        Показывает, как солвер хочет распределить потоки
        """
        directed = {}
        
        for inst in self.instances:
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                flow = inst.path_flows.get(path_key, 0.0)
                
                if flow <= 0.001:
                    continue
                
                # Восстанавливаем маршрут
                current_node = inst.request.source
                
                for edge in path:
                    # Определяем следующий узел
                    if edge.nodes[0] == current_node:
                        next_node = edge.nodes[1]
                    else:
                        next_node = edge.nodes[0]
                    
                    from_name = current_node.name
                    to_name = next_node.name
                    
                    key = (from_name, to_name)
                    directed[key] = directed.get(key, 0.0) + flow
                    
                    current_node = next_node
        
        return directed
    
    def get_edge_violations(self) -> List[Dict]:
        """Возвращает список рёбер с превышением пропускной способности."""
        loads = self.get_edge_loads()
        violations = []
        for edge, (flow, ratio) in loads.items():
            if edge.capacity != float('inf') and ratio > 1.0:
                violations.append({
                    'edge': f"{edge.nodes[0].name}-{edge.nodes[1].name}",
                    'capacity': edge.capacity,
                    'actual_flow': flow,
                    'excess': flow - edge.capacity,
                    'excess_pct': (ratio - 1.0) * 100
                })
        return violations

    def plot_training_history(self, filename="training_history.png"):
        """Строит график обучения: только train loss в логарифмическом масштабе"""
        if not self.loss_history:
            print("Нет истории обучения")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Train loss в логарифмическом масштабе
        iterations = range(len(self.loss_history))
        ax.semilogy(iterations, self.loss_history, 'b-', linewidth=1.5, label='Train Loss')
        
        ax.set_xlabel('Итерация')
        ax.set_ylabel('Loss (кВт, log scale)')
        ax.set_title('Функция потерь (логарифмический масштаб)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"График обучения сохранён в {filename}")

    def get_directed_edge_flows(self) -> Dict[Tuple[str, str], float]:
        """
        Возвращает направленные фактические потоки по рёбрам.
        
        Returns:
            {(from_node, to_node): flow} — фактические потоки после ограничений
        """
        # Получаем текущие оптимизируемые потоки
        current_flows = {}
        for inst_idx, inst in enumerate(self.instances):
            for path in inst.get_paths():
                path_key = inst._path_to_key(path)
                current_flows[(inst_idx, path_key)] = inst.path_flows[path_key]
        
        # Применяем ограничения — получаем фактические потоки
        actual_flows = self.compute_actual_flows(current_flows)
        
        # Суммируем направленные потоки
        directed = {}
        for (inst_idx, path_key), flow in actual_flows.items():
            if flow <= 0:
                continue
            
            inst = self.instances[inst_idx]
            for path in inst.get_paths():
                if inst._path_to_key(path) == path_key:
                    current_node = inst.request.source
                    for edge in path:
                        if edge.nodes[0] == current_node:
                            next_node = edge.nodes[1]
                        else:
                            next_node = edge.nodes[0]
                        
                        key = (current_node.name, next_node.name)
                        directed[key] = directed.get(key, 0.0) + flow
                        current_node = next_node
                    break
        
        return directed
