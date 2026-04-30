import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import qmc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DataGenerator:
    """
    Генератор обучающих данных с использованием Latin Hypercube Sampling
    
    Генерирует точки в единичном гиперкубе [0, 1]^(E + S*C)
    Часть capacity-признаков заменяется на inf согласно уровню разреженности
    Нормализация выполняется позже через FeatureExtractor.normalize_features()
    """
    def __init__(self, 
                 feature_dim: int,
                 sources: List[str],
                 consumers: List[str],
                 E: int):
        self.feature_dim = feature_dim
        self.sources = sources
        self.consumers = consumers
        self.E = E
        
        self.all_pairs = [(s, c) for s in sources for c in consumers]
    
    def generate_samples(self, 
                     num_samples: int,
                     sparsity_levels: List[float] = [0.3, 0.7],
                     demand_scale_factors: List[float] = [1.0]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Генерирует сценарии
        
        Args:
            num_samples: количество сэмплов на каждую комбинацию (sparsity, scale_factor)
            sparsity_levels: список долей рёбер, которые станут inf (0.3 = 30% inf)
            demand_scale_factors: список коэффициентов масштабирования demand-признаков
            
        Returns:
            raw_features: (total_samples, feature_dim) — значения в [0, 1] + inf
            demands_matrix: (total_samples, S, C) — заявки, масштабированные коэффициентом
            scenarios: список словарей заявок (для отладки)
        """
        all_features = []
        all_scenarios = []
        
        S = len(self.sources)
        C = len(self.consumers)
        
        for sparsity in sparsity_levels:
            for scale_factor in demand_scale_factors:
                print(f"\nГенерация для sparsity={sparsity:.2f} (доля inf capacity: {sparsity*100:.0f}%), "
                      f"demand_scale={scale_factor}")
                
                sampler = qmc.LatinHypercube(d=self.feature_dim)
                samples = sampler.random(n=num_samples)
                
                for i in range(num_samples):
                    row = samples[i].copy()
                    
                    # Заменяем часть capacity на inf согласно уровню разреженности
                    num_inf = int(self.E * sparsity)
                    if num_inf > 0:
                        inf_indices = np.random.choice(self.E, size=num_inf, replace=False)
                        row[inf_indices] = float('inf')
                    
                    # Масштабируем demand-признаки (последние S*C элементов)
                    row[self.E:] *= scale_factor
                    
                    all_features.append(row)
                    all_scenarios.append(self._row_to_flows(row))
        
        all_features = np.array(all_features, dtype=np.float32)
        
        # Формируем demands_matrix из тех же данных
        demands_matrix = np.zeros((len(all_features), S, C), dtype=np.float32)
        for i, row in enumerate(all_features):
            offset = self.E
            for idx, (s_name, c_name) in enumerate(self.all_pairs):
                s_idx = self.sources.index(s_name)
                c_idx = self.consumers.index(c_name)
                val = row[offset + idx]
                demands_matrix[i, s_idx, c_idx] = val
        
        total_combinations = len(sparsity_levels) * len(demand_scale_factors)
        print(f"\n✓ Сгенерировано {len(all_scenarios)} сценариев "
              f"({total_combinations} комбинаций × {num_samples} сэмплов)")
        print(f"  - Размер матрицы признаков: {all_features.shape}")
        print(f"  - Диапазон demands: [{demands_matrix.min():.4f}, {demands_matrix.max():.4f}]")
        
        return all_features, demands_matrix, all_scenarios
    
    def _row_to_flows(self, row: np.ndarray) -> Dict:
        """Преобразует строку признаков в словарь заявок (для отладки)"""
        flows = {s: {} for s in self.sources}
        offset = self.E
        for idx, (s_name, c_name) in enumerate(self.all_pairs):
            val = row[offset + idx]
            if val > 0:
                flows[s_name][c_name] = float(val)
        return flows


class DataVisualizer:
    """
    Визуализация сгенерированных данных с помощью PCA (слишком много информации теряется при таком сильном снижении размереости)
    """
    def __init__(self, save_dir: str = 'genereted'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_pca(self, 
                      features: np.ndarray,
                      labels: Optional[np.ndarray] = None,
                      save_name: str = 'pca_visualization.png',
                      show: bool = True) -> None:
        """
        Визуализирует данные в проекции на первые 3 главные компоненты PCA, inf заменяются на 0 для визуализации
        """
        features_clean = np.where(np.isfinite(features), features, 0.0)
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)
        
        pca = PCA(n_components=min(3, features.shape[1]))
        features_pca = pca.fit_transform(features_scaled)
        
        fig = plt.figure(figsize=(14, 10))
        
        # 3D проекция
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        xs = features_pca[:, 0].tolist()
        ys = features_pca[:, 1].tolist() if features_pca.shape[1] > 1 else np.zeros(len(xs)).tolist()
        zs = features_pca[:, 2].tolist() if features_pca.shape[1] > 2 else np.zeros(len(xs)).tolist()
        
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax1.plot(
                    np.array(xs)[mask], np.array(ys)[mask], np.array(zs)[mask],
                    'o', markersize=4, alpha=0.6, label=f'Класс {label}'
                )
            ax1.legend()
        else:
            ax1.plot(xs, ys, zs, 'o', markersize=4, alpha=0.6, color='blue')
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)' if features_pca.shape[1] > 1 else '')
        ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)' if features_pca.shape[1] > 2 else '')
        ax1.set_title('3D PCA проекция')
        
        # 2D проекция
        ax2 = fig.add_subplot(2, 2, 2)
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax2.plot(features_pca[mask, 0], 
                        features_pca[mask, 1] if features_pca.shape[1] > 1 else np.zeros(mask.sum()),
                        'o', markersize=4, alpha=0.6, label=f'Класс {label}')
            ax2.legend()
        else:
            ax2.plot(features_pca[:, 0], 
                    features_pca[:, 1] if features_pca.shape[1] > 1 else np.zeros(len(features_pca)),
                    'o', markersize=4, alpha=0.6, color='blue')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)' if features_pca.shape[1] > 1 else '')
        ax2.set_title('PC1 vs PC2')
        ax2.grid(True, alpha=0.3)
        
        if features_pca.shape[1] > 2:
            ax3 = fig.add_subplot(2, 2, 3)
            if labels is not None:
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    mask = labels == label
                    ax3.plot(features_pca[mask, 0], features_pca[mask, 2],
                            'o', markersize=4, alpha=0.6, label=f'Класс {label}')
                ax3.legend()
            else:
                ax3.plot(features_pca[:, 0], features_pca[:, 2],
                        'o', markersize=4, alpha=0.6, color='blue')
            ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax3.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
            ax3.set_title('PC1 vs PC3')
            ax3.grid(True, alpha=0.3)
        
        # Объяснённая дисперсия
        ax4 = fig.add_subplot(2, 2, 4)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        ax4.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-o', markersize=4)
        ax4.axhline(y=0.9, color='r', linestyle='--', label='90% дисперсии')
        ax4.axhline(y=0.95, color='orange', linestyle='--', label='95% дисперсии')
        ax4.set_xlabel('Количество компонент')
        ax4.set_ylabel('Накопленная объяснённая дисперсия')
        ax4.set_title('Объяснённая дисперсия PCA')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f"\n !PCA визуализация сохранена в {save_path}!")
        print(f"  - PC1 объясняет {pca.explained_variance_ratio_[0]*100:.1f}% дисперсии")
    
    def visualize_distribution(self,
                               features: np.ndarray,
                               save_name: str = 'distribution.png',
                               show: bool = True) -> None:
        """
        Визуализирует распределение значений признаков
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Распределение средних значений признаков (только конечных)
        ax = axes[0]
        finite_features = np.where(np.isfinite(features), features, np.nan)
        feature_means = np.nanmean(finite_features, axis=0)
        feature_means = feature_means[np.isfinite(feature_means)]
        
        ax.hist(feature_means, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Среднее значение признака')
        ax.set_ylabel('Количество признаков')
        ax.set_title('Распределение средних значений признаков (без inf)')
        ax.axvline(x=np.mean(feature_means), color='r', linestyle='--',
                  label=f'Среднее: {np.mean(feature_means):.4e}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Распределение всех конечных значений
        ax = axes[1]
        all_finite = features[np.isfinite(features)]
        ax.hist(all_finite, bins=100, edgecolor='black', alpha=0.7, density=True)
        ax.set_xlabel('Значение признака')
        ax.set_ylabel('Плотность')
        ax.set_title('Распределение конечных значений признаков')
        
        stats_text = (f"Статистика:\n"
                     f"Мин: {all_finite.min():.4e}\n"
                     f"Макс: {all_finite.max():.4e}\n"
                     f"Медиана: {np.median(all_finite):.4e}\n"
                     f"Доля inf: {(~np.isfinite(features)).mean():.2%}")
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)
        
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f"\n !Распределение признаков сохранено в {save_path}!")