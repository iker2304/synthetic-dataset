#!/usr/bin/env python3
"""
Script completo para entrenar un modelo YOLO utilizando PyTorch
Autor: Generado automáticamente
Fecha: 2024

Este script incluye:
- Configuración del dataset con conversión de JSON a formato YOLO
- División 80/20 para entrenamiento/validación
- Preprocesamiento y data augmentation
- Entrenamiento con early stopping
- Monitoreo de métricas y visualización
- Manejo de errores completo
"""

import os
import sys
import json
import shutil
import logging
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Configurar warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class YOLOConfig:
    """Configuración de hiperparámetros para el entrenamiento YOLO"""
    
    def __init__(self, task_type="detection"):
        # Tipo de tarea: "detection" o "segmentation"
        self.task_type = task_type
        
        # Rutas del dataset
        self.dataset_path = r"C:\Users\ikerc\Documents\UPC\TFG\Software\synthetic-dataset-cube\render_output"
        self.annotations_path = os.path.join(self.dataset_path, "annotations")
        self.images_path = self.dataset_path
        
        # Configuración del modelo según el tipo de tarea
        if task_type == "segmentation":
            self.model_name = "yolov8n-seg"  # YOLOv8 nano segmentation
            self.task_suffix = "_seg"
        else:
            self.model_name = "yolov8n"  # YOLOv8 nano detection
            self.task_suffix = ""
            
        self.img_size = 640
        self.num_classes = 1  # Ajustar según el dataset
        
        # Hiperparámetros de entrenamiento
        self.batch_size = 32
        self.epochs = 120
        self.learning_rate = 0.01
        self.momentum = 0.937
        self.weight_decay = 0.0005
        self.patience = 50  # Para early stopping
        
        # División del dataset
        self.train_split = 0.8
        self.val_split = 0.2
        
        # Configuración de data augmentation
        self.augmentation = {
            'horizontal_flip': 0.5,
            'saturation_range': (0.7, 1.3),
            'brightness_range': (0.8, 1.2),
            'scale_range': (0.8, 1.2),
            'rotation_range': (-10, 10)
        }
        
        # Directorios de salida
        self.output_dir = f"yolo_training_output{self.task_suffix}"
        self.models_dir = os.path.join(self.output_dir, "models")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.logs_dir = os.path.join(self.output_dir, "logs")

class DatasetConverter:
    """Convierte anotaciones JSON a formato YOLO"""
    
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.class_names = {}
        self.class_count = 0
        
    def json_to_yolo_format(self, json_annotation: Dict, img_width: int, img_height: int) -> List[str]:
        """
        Convierte una anotación JSON al formato YOLO
        
        Args:
            json_annotation: Diccionario con la anotación JSON
            img_width: Ancho de la imagen
            img_height: Alto de la imagen
            
        Returns:
            Lista de strings en formato YOLO
        """
        yolo_annotations = []
        
        if 'objects' not in json_annotation:
            return yolo_annotations
            
        for obj in json_annotation['objects']:
            # Obtener bounding box
            bbox = obj.get('bbox_2d', {})
            if not bbox:
                continue
                
            # Manejar diferentes formatos de coordenadas
            x_min = bbox.get('min_x', bbox.get('x_min', 0))
            y_min = bbox.get('min_y', bbox.get('y_min', 0))
            x_max = bbox.get('max_x', bbox.get('x_max', 0))
            y_max = bbox.get('max_y', bbox.get('y_max', 0))
            
            # Obtener clase del objeto
            object_name = obj.get('object_name', 'unknown')
            if object_name not in self.class_names:
                self.class_names[object_name] = self.class_count
                self.class_count += 1
                
            class_id = self.class_names[object_name]
            
            if self.config.task_type == "detection":
                # Convertir a formato YOLO para detección (coordenadas normalizadas)
                center_x = (x_min + x_max) / 2.0 / img_width
                center_y = (y_min + y_max) / 2.0 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                # Formato YOLO: class_id center_x center_y width height
                yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                yolo_annotations.append(yolo_line)
                
            elif self.config.task_type == "segmentation":
                # Generar máscara sintética a partir del bounding box
                # Para segmentación, creamos un polígono que aproxima la forma del objeto
                # Usamos un rectángulo con esquinas ligeramente redondeadas
                
                # Normalizar coordenadas
                x_min_norm = x_min / img_width
                y_min_norm = y_min / img_height
                x_max_norm = x_max / img_width
                y_max_norm = y_max / img_height
                
                # Crear polígono rectangular (8 puntos para simular esquinas redondeadas)
                margin = 0.02  # Pequeño margen para simular forma más realista
                
                # Puntos del polígono (en sentido horario)
                polygon_points = [
                    x_min_norm + margin, y_min_norm,  # Top-left
                    x_max_norm - margin, y_min_norm,  # Top-right
                    x_max_norm, y_min_norm + margin,  # Right-top
                    x_max_norm, y_max_norm - margin,  # Right-bottom
                    x_max_norm - margin, y_max_norm,  # Bottom-right
                    x_min_norm + margin, y_max_norm,  # Bottom-left
                    x_min_norm, y_max_norm - margin,  # Left-bottom
                    x_min_norm, y_min_norm + margin   # Left-top
                ]
                
                # Asegurar que los puntos estén dentro de los límites [0, 1]
                polygon_points = [max(0.0, min(1.0, point)) for point in polygon_points]
                
                # Formato YOLO segmentación: class_id x1 y1 x2 y2 x3 y3 ...
                polygon_str = ' '.join([f"{point:.6f}" for point in polygon_points])
                yolo_line = f"{class_id} {polygon_str}"
                yolo_annotations.append(yolo_line)
            
        return yolo_annotations
    
    def convert_dataset(self) -> Tuple[List[str], List[str]]:
        """
        Convierte todo el dataset de JSON a formato YOLO
        
        Returns:
            Tupla con listas de rutas de imágenes y anotaciones
        """
        logger.info("Iniciando conversión del dataset...")
        
        image_paths = []
        annotation_paths = []
        
        # Crear directorio para anotaciones YOLO
        yolo_annotations_dir = os.path.join(self.config.output_dir, "yolo_annotations")
        os.makedirs(yolo_annotations_dir, exist_ok=True)
        
        # Procesar cada archivo de anotación
        annotation_files = [f for f in os.listdir(self.config.annotations_path) if f.endswith('.json')]
        
        for ann_file in annotation_files:
            try:
                # Cargar anotación JSON
                json_path = os.path.join(self.config.annotations_path, ann_file)
                with open(json_path, 'r') as f:
                    annotation = json.load(f)
                
                # Obtener nombre de imagen correspondiente
                image_filename = annotation.get('image_filename', '')
                if not image_filename:
                    continue
                    
                image_path = os.path.join(self.config.images_path, image_filename)
                if not os.path.exists(image_path):
                    logger.warning(f"Imagen no encontrada: {image_path}")
                    continue
                
                # Obtener dimensiones de la imagen
                img_resolution = annotation.get('image_resolution', {})
                img_width = img_resolution.get('width', 1920)
                img_height = img_resolution.get('height', 1080)
                
                # Convertir a formato YOLO
                yolo_annotations = self.json_to_yolo_format(annotation, img_width, img_height)
                
                # Guardar anotación YOLO
                yolo_filename = ann_file.replace('_annotations.json', '.txt')
                yolo_path = os.path.join(yolo_annotations_dir, yolo_filename)
                
                with open(yolo_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                image_paths.append(image_path)
                annotation_paths.append(yolo_path)
                
            except Exception as e:
                logger.error(f"Error procesando {ann_file}: {str(e)}")
                continue
        
        # Guardar nombres de clases
        classes_file = os.path.join(self.config.output_dir, "classes.txt")
        with open(classes_file, 'w') as f:
            for class_name, class_id in sorted(self.class_names.items(), key=lambda x: x[1]):
                f.write(f"{class_name}\n")
        
        logger.info(f"Conversión completada. {len(image_paths)} imágenes procesadas.")
        logger.info(f"Clases detectadas: {list(self.class_names.keys())}")
        
        return image_paths, annotation_paths

class YOLODataset(Dataset):
    """Dataset personalizado para YOLO"""
    
    def __init__(self, image_paths: List[str], annotation_paths: List[str], 
                 img_size: int = 640, augment: bool = False, config: YOLOConfig = None):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.img_size = img_size
        self.augment = augment
        self.config = config
        
        # Transformaciones base
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Cargar imagen
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Cargar anotaciones
        annotation_path = self.annotation_paths[idx]
        boxes = []
        labels = []
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            center_x = float(parts[1])
                            center_y = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convertir de formato YOLO a coordenadas absolutas
                            x1 = (center_x - width/2) * self.img_size
                            y1 = (center_y - height/2) * self.img_size
                            x2 = (center_x + width/2) * self.img_size
                            y2 = (center_y + height/2) * self.img_size
                            
                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_id)
        
        # Aplicar data augmentation si está habilitado
        if self.augment and self.config:
            image, boxes = self.apply_augmentation(image, boxes)
        
        # Aplicar transformaciones
        image = self.base_transform(image)
        
        # Convertir a tensores
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        return image, target
    
    def apply_augmentation(self, image, boxes):
        """Aplica data augmentation a la imagen y ajusta las bounding boxes"""
        
        # Flip horizontal aleatorio
        if random.random() < self.config.augmentation['horizontal_flip']:
            image = transforms.functional.hflip(image)
            # Ajustar boxes para flip horizontal
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                boxes[i] = [self.img_size - x2, y1, self.img_size - x1, y2]
        
        # Ajustes de color
        saturation_factor = random.uniform(*self.config.augmentation['saturation_range'])
        brightness_factor = random.uniform(*self.config.augmentation['brightness_range'])
        
        image = transforms.functional.adjust_saturation(image, saturation_factor)
        image = transforms.functional.adjust_brightness(image, brightness_factor)
        
        return image, boxes

class YOLOTrainer:
    """Clase principal para el entrenamiento del modelo YOLO"""
    
    def __init__(self, config: YOLOConfig):
        self.config = config
        
        # FORZAR USO DE CUDA - No permitir CPU
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA no está disponible. Este script requiere GPU para entrenamiento.\n"
                "Verifica que:\n"
                "1. Tienes una GPU NVIDIA compatible\n"
                "2. Los drivers de NVIDIA están instalados\n"
                "3. PyTorch con soporte CUDA está instalado\n"
                "Instala PyTorch con CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            )
        
        self.device = torch.device('cuda')
        
        # Información detallada de la GPU
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        logger.info(f"FORZANDO USO DE GPU: {gpu_name}")
        logger.info(f"Memoria GPU disponible: {gpu_memory:.2f} GB")
        
        # Verificar memoria mínima recomendada
        if gpu_memory < 4.0:
            logger.warning(f"Memoria GPU baja ({gpu_memory:.2f} GB). Se recomienda al menos 4GB para YOLO")
            logger.warning("Considera reducir batch_size si encuentras errores de memoria")
        
        # Limpiar caché de GPU al inicio
        torch.cuda.empty_cache()
        logger.info("Caché de GPU limpiado")
        
        # Crear directorios de salida
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.models_dir, exist_ok=True)
        os.makedirs(self.config.plots_dir, exist_ok=True)
        os.makedirs(self.config.logs_dir, exist_ok=True)
        
        # Métricas de entrenamiento
        self.train_losses = []
        self.val_losses = []
        self.val_maps = []
        self.best_map = 0.0
        self.patience_counter = 0
        
    def setup_model(self):
        """Configura el modelo YOLO"""
        try:
            # Intentar importar ultralytics
            from ultralytics import YOLO
            
            # Crear modelo YOLO
            self.model = YOLO(f'{self.config.model_name}.pt')
            logger.info(f"Modelo {self.config.model_name} cargado exitosamente")
            
        except ImportError:
            logger.error("ultralytics no está instalado. Instalando...")
            os.system("pip install ultralytics")
            from ultralytics import YOLO
            self.model = YOLO(f'{self.config.model_name}.pt')
            
        except Exception as e:
            logger.error(f"Error configurando el modelo: {str(e)}")
            raise
    
    def prepare_dataset(self):
        """Prepara el dataset para entrenamiento"""
        logger.info("Preparando dataset...")
        
        # Convertir anotaciones
        converter = DatasetConverter(self.config)
        image_paths, annotation_paths = converter.convert_dataset()
        
        if len(image_paths) == 0:
            raise ValueError("No se encontraron imágenes válidas en el dataset")
        
        # Dividir dataset
        train_images, val_images, train_annotations, val_annotations = train_test_split(
            image_paths, annotation_paths, 
            test_size=self.config.val_split, 
            random_state=42,
            shuffle=True
        )
        
        logger.info(f"Dataset dividido: {len(train_images)} entrenamiento, {len(val_images)} validación")
        
        # Crear datasets
        self.train_dataset = YOLODataset(
            train_images, train_annotations, 
            self.config.img_size, augment=True, config=self.config
        )
        
        self.val_dataset = YOLODataset(
            val_images, val_annotations, 
            self.config.img_size, augment=False, config=self.config
        )
        
        # Crear data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=4,
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=4,
            collate_fn=self.collate_fn
        )
        
        # Actualizar número de clases
        self.config.num_classes = converter.class_count
        
    def collate_fn(self, batch):
        """Función personalizada para agrupar muestras del dataset"""
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        images = torch.stack(images, 0)
        return images, targets
    
    def create_yolo_config_file(self):
        """Crea archivo de configuración para YOLO"""
        
        # Crear estructura de directorios para YOLO
        yolo_dataset_dir = os.path.join(self.config.output_dir, "yolo_dataset")
        train_images_dir = os.path.join(yolo_dataset_dir, "images", "train")
        val_images_dir = os.path.join(yolo_dataset_dir, "images", "val")
        train_labels_dir = os.path.join(yolo_dataset_dir, "labels", "train")
        val_labels_dir = os.path.join(yolo_dataset_dir, "labels", "val")
        
        for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Copiar imágenes y etiquetas a la estructura YOLO
        logger.info("Organizando dataset en estructura YOLO...")
        
        # Entrenamiento
        for i, (img_path, ann_path) in enumerate(zip(self.train_dataset.image_paths, self.train_dataset.annotation_paths)):
            img_name = os.path.basename(img_path)
            ann_name = os.path.basename(ann_path)
            
            shutil.copy2(img_path, os.path.join(train_images_dir, img_name))
            shutil.copy2(ann_path, os.path.join(train_labels_dir, ann_name))
        
        # Validación
        for i, (img_path, ann_path) in enumerate(zip(self.val_dataset.image_paths, self.val_dataset.annotation_paths)):
            img_name = os.path.basename(img_path)
            ann_name = os.path.basename(ann_path)
            
            shutil.copy2(img_path, os.path.join(val_images_dir, img_name))
            shutil.copy2(ann_path, os.path.join(val_labels_dir, ann_name))
        
        # Crear archivo de configuración YAML
        config_yaml = f"""
# Dataset configuration for YOLO training
path: {yolo_dataset_dir}
train: images/train
val: images/val

# Classes
nc: {self.config.num_classes}
names: {list(range(self.config.num_classes))}
"""
        
        config_file = os.path.join(self.config.output_dir, "dataset.yaml")
        with open(config_file, 'w') as f:
            f.write(config_yaml)
        
        return config_file
    
    def train(self):
        """Ejecuta el entrenamiento del modelo"""
        logger.info("Iniciando entrenamiento con GPU forzada...")
        
        try:
            # Preparar dataset
            self.prepare_dataset()
            
            # Configurar modelo
            self.setup_model()
            
            # Crear archivo de configuración
            config_file = self.create_yolo_config_file()
            
            # Verificar estado de GPU antes del entrenamiento
            logger.info(f"Estado GPU antes del entrenamiento:")
            logger.info(f"   - Memoria GPU usada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            logger.info(f"   - Memoria GPU reservada: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            
            # Entrenar modelo usando ultralytics FORZANDO CUDA
            results = self.model.train(
                data=config_file,
                epochs=self.config.epochs,
                imgsz=self.config.img_size,
                batch=self.config.batch_size,
                lr0=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                patience=self.config.patience,
                save=True,
                project=self.config.output_dir,
                name='yolo_training',
                exist_ok=True,
                plots=True,
                device=0  # Forzar GPU 0 (primera GPU disponible)
            )
            
            logger.info("Entrenamiento completado exitosamente en GPU")
            
            # Mostrar estadísticas finales de GPU
            logger.info(f"Estado final GPU:")
            logger.info(f"   - Memoria GPU máxima usada: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
            logger.info(f"   - Memoria GPU actual: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            
            # Guardar modelo final
            final_model_path = os.path.join(self.config.models_dir, "best_model.pt")
            self.model.save(final_model_path)
            
            # Generar reportes
            self.generate_training_report(results)
            
            return results
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error("Error de memoria GPU insuficiente!")
            logger.error("Soluciones recomendadas:")
            logger.error("   1. Reducir batch_size en YOLOConfig")
            logger.error("   2. Reducir img_size (ej: 416 en lugar de 640)")
            logger.error("   3. Usar modelo más pequeño (yolov8n en lugar de yolov8s/m/l)")
            logger.error("   4. Cerrar otras aplicaciones que usen GPU")
            raise
            
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {str(e)}")
            raise
    
    def generate_training_report(self, results):
        """Genera reporte completo del entrenamiento"""
        logger.info("Generando reporte de entrenamiento...")
        
        try:
            # Crear gráficas de métricas
            self.plot_training_metrics(results)
            
            # Generar reporte de texto
            report_path = os.path.join(self.config.output_dir, "training_report.txt")
            with open(report_path, 'w') as f:
                f.write("=== REPORTE DE ENTRENAMIENTO YOLO ===\n\n")
                f.write(f"Configuración del modelo:\n")
                f.write(f"- Modelo: {self.config.model_name}\n")
                f.write(f"- Épocas: {self.config.epochs}\n")
                f.write(f"- Batch size: {self.config.batch_size}\n")
                f.write(f"- Learning rate: {self.config.learning_rate}\n")
                f.write(f"- Tamaño de imagen: {self.config.img_size}\n")
                f.write(f"- Número de clases: {self.config.num_classes}\n\n")
                
                f.write(f"Dataset:\n")
                f.write(f"- Imágenes de entrenamiento: {len(self.train_dataset)}\n")
                f.write(f"- Imágenes de validación: {len(self.val_dataset)}\n\n")
                
                if hasattr(results, 'results_dict'):
                    f.write("Métricas finales:\n")
                    for key, value in results.results_dict.items():
                        f.write(f"- {key}: {value}\n")
            
            logger.info(f"Reporte guardado en: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generando reporte: {str(e)}")
    
    def plot_training_metrics(self, results):
        """Genera gráficas de las métricas de entrenamiento"""
        try:
            # Configurar estilo de gráficas
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Métricas de Entrenamiento YOLO', fontsize=16)
            
            # Nota: Las métricas específicas dependen de la versión de ultralytics
            # Aquí se muestra un ejemplo genérico
            
            # Gráfica de pérdida (ejemplo)
            axes[0, 0].set_title('Pérdida de Entrenamiento')
            axes[0, 0].set_xlabel('Época')
            axes[0, 0].set_ylabel('Pérdida')
            axes[0, 0].grid(True)
            
            # Gráfica de mAP (ejemplo)
            axes[0, 1].set_title('mAP@0.5')
            axes[0, 1].set_xlabel('Época')
            axes[0, 1].set_ylabel('mAP')
            axes[0, 1].grid(True)
            
            # Gráfica de precisión (ejemplo)
            axes[1, 0].set_title('Precisión')
            axes[1, 0].set_xlabel('Época')
            axes[1, 0].set_ylabel('Precisión')
            axes[1, 0].grid(True)
            
            # Gráfica de recall (ejemplo)
            axes[1, 1].set_title('Recall')
            axes[1, 1].set_xlabel('Época')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Guardar gráfica
            plot_path = os.path.join(self.config.plots_dir, 'training_metrics.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Gráficas guardadas en: {plot_path}")
            
        except Exception as e:
            logger.error(f"Error generando gráficas: {str(e)}")

def validate_dataset_path(dataset_path: str) -> bool:
    """Valida que la ruta del dataset existe y contiene los archivos necesarios"""
    
    if not os.path.exists(dataset_path):
        logger.error(f"La ruta del dataset no existe: {dataset_path}")
        return False
    
    annotations_path = os.path.join(dataset_path, "annotations")
    if not os.path.exists(annotations_path):
        logger.error(f"Directorio de anotaciones no encontrado: {annotations_path}")
        return False
    
    # Verificar que hay archivos de anotación
    annotation_files = [f for f in os.listdir(annotations_path) if f.endswith('.json')]
    if len(annotation_files) == 0:
        logger.error("No se encontraron archivos de anotación JSON")
        return False
    
    # Verificar que hay imágenes correspondientes
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.png')]
    if len(image_files) == 0:
        logger.error("No se encontraron archivos de imagen PNG")
        return False
    
    logger.info(f"Dataset validado: {len(annotation_files)} anotaciones, {len(image_files)} imágenes")
    return True

def check_memory_requirements():
    """Verifica los requisitos de memoria del sistema"""
    try:
        import psutil
        
        # Obtener información de memoria
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        logger.info(f"Memoria disponible: {available_gb:.2f} GB")
        
        if available_gb < 4.0:
            logger.warning("Memoria disponible baja. Se recomienda al menos 4GB para entrenamiento YOLO")
            return False
        
        return True
        
    except ImportError:
        logger.warning("psutil no disponible. No se puede verificar memoria")
        return True
    except Exception as e:
        logger.error(f"Error verificando memoria: {str(e)}")
        return True

def main():
    """Función principal del script"""
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenamiento YOLO para detección y segmentación')
    parser.add_argument('--task', type=str, choices=['detection', 'segmentation'], 
                       default='detection', help='Tipo de tarea: detection o segmentation')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=16, help='Tamaño del batch')
    parser.add_argument('--img-size', type=int, default=640, help='Tamaño de imagen')
    
    args = parser.parse_args()
    
    logger.info(f"=== INICIANDO ENTRENAMIENTO YOLO {args.task.upper()} CON GPU FORZADA ===")
    
    try:
        # Verificación OBLIGATORIA de CUDA antes de continuar
        if not torch.cuda.is_available():
            error_msg = (
                "CUDA NO DISPONIBLE - ENTRENAMIENTO CANCELADO\n"
                "Este script requiere GPU NVIDIA para funcionar.\n\n"
                "Pasos para solucionar:\n"
                "1. Verifica que tienes una GPU NVIDIA compatible\n"
                "2. Instala los drivers de NVIDIA más recientes\n"
                "3. Instala PyTorch con soporte CUDA:\n"
                "   pip uninstall torch torchvision\n"
                "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n"
                "4. Reinicia el sistema después de instalar drivers\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Información detallada de GPU disponible
        gpu_count = torch.cuda.device_count()
        logger.info(f"GPUs detectadas: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # Seleccionar GPU principal
        torch.cuda.set_device(0)
        logger.info(f"Usando GPU 0: {torch.cuda.get_device_name(0)}")
        
        # Crear configuración con el tipo de tarea especificado
        config = YOLOConfig(task_type=args.task)
        
        # Aplicar argumentos de línea de comandos a la configuración
        config.epochs = args.epochs
        config.batch_size = args.batch_size
        config.img_size = args.img_size
        
        # Validaciones previas
        logger.info("Realizando validaciones del sistema...")
        
        # Validar dataset
        if not validate_dataset_path(config.dataset_path):
            raise ValueError("Validación del dataset falló")
        
        # Verificar memoria GPU específicamente
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        recommended_batch_size = min(16, max(4, int(gpu_memory / 2)))
        
        if gpu_memory < 4.0:
            logger.warning(f"GPU con poca memoria ({gpu_memory:.2f} GB)")
            logger.warning(f"Ajustando batch_size automáticamente a {recommended_batch_size}")
            config.batch_size = recommended_batch_size
        
        logger.info(f"Memoria GPU: {gpu_memory:.2f} GB")
        logger.info(f"Batch size configurado: {config.batch_size}")
        
        # Crear entrenador (esto ya verificará CUDA internamente)
        trainer = YOLOTrainer(config)
        
        # Ejecutar entrenamiento
        results = trainer.train()
        
        logger.info("=== ENTRENAMIENTO COMPLETADO EXITOSAMENTE EN GPU ===")
        logger.info(f"Resultados guardados en: {config.output_dir}")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("Entrenamiento interrumpido por el usuario")
        return None
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.error("Error relacionado con CUDA/GPU")
            logger.error("Revisa la instalación de drivers y PyTorch con CUDA")
        logger.error(f"Error: {str(e)}")
        return None
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        logger.error("Traceback completo:", exc_info=True)
        return None

if __name__ == "__main__":
    # Configurar semilla para reproducibilidad
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # Para múltiples GPUs
    
    # Ejecutar entrenamiento
    results = main()
    
    if results is not None:
        print("\n¡Entrenamiento completado exitosamente en GPU!")
        print("Revisa los archivos de salida para ver los resultados.")
        print("El modelo ha sido entrenado usando aceleración GPU.")
    else:
        print("\nEl entrenamiento no se completó correctamente.")
        print("Revisa los logs para más información.")
        print("Asegúrate de tener CUDA instalado y una GPU compatible.")