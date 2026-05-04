import copy
import os
import json

from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np

import matplotlib.pyplot as plt

from pyvis.network import Network

class TrainingVisualizer:
    """
    Визуализация процесса обучения нейросети
    """
    
    def __init__(self, save_dir: str = 'genereted'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_history(self, 
                            history: Dict,
                            save_name: str = 'training_history.png',
                            show: bool = True) -> None:
        """
        Строит графики обучения: loss и компоненты
        
        Args:
            history: словарь с историей обучения из ModelTrainer
            save_name: имя файла для сохранения
            show: показать ли график
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(len(history['train_loss']))
        
        # общий лосс
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Loss')
        ax.set_title('Общая функция потерь')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # отмечаем минимум валидационного лосса
        if history['val_loss']:
            min_val_epoch = np.argmin(history['val_loss'])
            min_val_loss = history['val_loss'][min_val_epoch]
            ax.plot(min_val_epoch, min_val_loss, 'r*', markersize=10, 
                   label=f'Мин. val: {min_val_loss:.4f}')
            ax.legend()
        
        # компоненты лосса на train
        ax = axes[0, 1]
        if history['train_components']:
            # Собираем данные по компонентам
            components = {}
            for comp_dict in history['train_components']:
                for key, value in comp_dict.items():
                    if key not in components:
                        components[key] = []
                    components[key].append(value)
            
            for key, values in components.items():
                ax.plot(epochs, values, label=key, linewidth=1.5)
            
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Значение компонента')
            ax.set_title('Компоненты лосса (Train)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # компоненты лосса на validation
        ax = axes[1, 0]
        if history['val_components']:
            components = {}
            for comp_dict in history['val_components']:
                for key, value in comp_dict.items():
                    if key not in components:
                        components[key] = []
                    components[key].append(value)
            
            for key, values in components.items():
                ax.plot(epochs, values, label=key, linewidth=1.5)
            
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Значение компонента')
            ax.set_title('Компоненты лосса (Validation)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # дополнительные метрики
        ax = axes[1, 1]
        if history['val_components']:
            # Достаём delivery_ratio и avg_utilization
            delivery_ratios = []
            avg_utils = []
            for comp_dict in history['val_components']:
                if 'delivery_ratio' in comp_dict:
                    delivery_ratios.append(comp_dict['delivery_ratio'] * 100)
                if 'avg_utilization' in comp_dict:
                    avg_utils.append(comp_dict['avg_utilization'] * 100)
            
            if delivery_ratios:
                ax.plot(epochs, delivery_ratios, 'g-', label='Доставка (%)', linewidth=2)
            if avg_utils:
                ax.plot(epochs, avg_utils, 'orange', label='Загрузка рёбер (%)', linewidth=2)
            
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Проценты')
            ax.set_title('Метрики качества')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 105])
        
        plt.tight_layout()
        
        # сохраняем
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Графики обучения сохранены в {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_learning_curves_comparison(self,
                                        histories: List[Dict],
                                        labels: List[str],
                                        save_name: str = 'learning_curves_comparison.png',
                                        show: bool = True) -> None:
        """
        Сравнивает кривые обучения для разных запусков (полезно при подборе гиперпараметров)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Train loss
        ax = axes[0]
        for history, label in zip(histories, labels):
            epochs = range(len(history['train_loss']))
            ax.plot(epochs, history['train_loss'], label=label, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Loss')
        ax.set_title('Train Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation loss
        ax = axes[1]
        for history, label in zip(histories, labels):
            epochs = range(len(history['val_loss']))
            ax.plot(epochs, history['val_loss'], label=label, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
