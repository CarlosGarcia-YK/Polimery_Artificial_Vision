from optparse import Values
import sys
import os
import cv2
from matplotlib import path
import skimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from pandas import value_counts
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
import tempfile
from tqdm import tqdm
from PIL import Image
import time 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog, QSlider, QHBoxLayout,QComboBox, QListWidgetItem,
    QVBoxLayout, QSplitter, QListWidget, QFormLayout, QSpinBox, QMessageBox, QProgressBar,QAction, qApp, QStackedWidget, QFileDialog, QMessageBox, QPushButton, QFrame, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer , QPoint, QRect
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QPolygon, QColor, QBrush, QIcon

import shutil
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import os

#------------------QLabel para la seleccion en la imagen--------------------------

class LassoLabel(QLabel):
   
    selectionFinished = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.selection_enabled = False  # Flag to enable/disable selection mode.
        self.drawing = False            # True while drawing the lasso.
        self.points = []                # List of points (in widget coordinates) for the current polygon.
        self.polygons = []              # List of finished polygons (each is a list of QPoint in widget coordinates).
        self._pixmapRect = None         # The rectangle (inside the label) where the image is drawn.
        self._scale_factor = 1.0        # Scale factor used to display the image.

    def enableSelection(self):
        """Enable lasso selection mode."""
        self.selection_enabled = True

    def clearPolygons(self):
        """Clear all drawn polygons."""
        self.polygons = []
        self.points = []
        self.update()

    def setPixmap(self, pixmap):
        """Override setPixmap so we can compute the drawing rectangle for the image."""
        super().setPixmap(pixmap)
        self._computePixmapRect()
        self.update()

    def _computePixmapRect(self):
     
        if self.pixmap() is None:
            self._pixmapRect = None
            return

        pixmap_size = self.pixmap().size()
        label_size = self.size()
        # Compute scale factor to fit the image inside the label.
        scale = min(label_size.width() / pixmap_size.width(),
                    label_size.height() / pixmap_size.height())
        new_width = pixmap_size.width() * scale
        new_height = pixmap_size.height() * scale
        x = (label_size.width() - new_width) / 2
        y = (label_size.height() - new_height) / 2
        self._pixmapRect = QRect(int(x), int(y), int(new_width), int(new_height))
        self._scale_factor = scale

    def mousePressEvent(self, event):
        if not self.selection_enabled or self.pixmap() is None:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.LeftButton and self._pixmapRect and self._pixmapRect.contains(event.pos()):
            self.drawing = True
            self.points = [event.pos()]
            self.update()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.selection_enabled:
            pos = event.pos()
            # Clamp the position to the pixmap rectangle if needed.
            if self._pixmapRect:
                x = max(self._pixmapRect.left(), min(pos.x(), self._pixmapRect.right()))
                y = max(self._pixmapRect.top(), min(pos.y(), self._pixmapRect.bottom()))
                pos = QPoint(x, y)
            self.points.append(pos)
            self.update()
        else:
            super().mouseMoveEvent(event) 

    def mouseReleaseEvent(self, event):
        if self.drawing and self.selection_enabled and event.button() == Qt.LeftButton:
            self.drawing = False
            # Optionally close the polygon (i.e. add the first point again if needed)
            if self.points and (self.points[0] != self.points[-1]):
                self.points.append(self.points[0])
            # Save the finished polygon (in widget coordinates) for display.
            self.polygons.append(self.points.copy())

            image_polygon = []
            if self._pixmapRect:
                for point in self.points:
                    image_x = (point.x() - self._pixmapRect.x()) / self._scale_factor
                    image_y = (point.y() - self._pixmapRect.y()) / self._scale_factor
                    image_polygon.append((image_x, image_y))  # O convertir a int según se requiera
            self.selectionFinished.emit(image_polygon)
            self.points = []

            self.update()
        else:
            super().mouseReleaseEvent(event)

    def paintEvent(self, event):
      
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw finished polygons (with fill).
        for poly in self.polygons:
            qpoly = QPolygon(poly)
            painter.setPen(QPen(Qt.red, 2))
            painter.setBrush(QBrush(QColor(255, 0, 0, 100)))  # Semi-transparent red fill.
            painter.drawPolygon(qpoly)

        # Draw the current polygon being drawn.
        if self.drawing and self.points:
            qpoly = QPolygon(self.points)
            painter.setPen(QPen(Qt.blue, 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolyline(qpoly)
        
   

# ----------------- QTHREAD PARA CADA PAGINA -------------------

class ImageProcessor(QThread):
    # Señales para comunicación con la GUI
    progress_updated = pyqtSignal(int, float, float) # Progreso del procesamiento
    image_processed = pyqtSignal(np.ndarray, int)  # Imagen resultante y contador
    batch_finished = pyqtSignal(pd.DataFrame)      # Resultados finales del lote
    error_occurred = pyqtSignal(str)  
    finished = pyqtSignal(bool)  # Señal de finalización del procesamiento
    def __init__(self, image_paths, params):
        super().__init__()
        self.image_paths = image_paths
        self.params = params
        self.is_running = True  # Bandera para controlar la ejecución
        self.total_stages = 11  # Número total de etapas
        self.avg_time_per_image = 0
        

    def run(self):
        try:
            self.start_time = time.time()
            total_images = len(self.image_paths)
            processed_images = 0
            all_batch_data = []  # Inicializar lista para acumular datos
            total_images = len(self.image_paths)
            for idx, img_path in enumerate(self.image_paths):
                if not self.is_running:
                    break
                
                # Procesar imagen y obtener resultados
                if total_images > 1:
                    result, _, _, batch_data = self.process_batch(img_path, self.params)
                if total_images == 1:
                    result, valid_count, _, batch_data = self.process_batch(img_path, self.params)
                valid_count = len(batch_data) if batch_data else 0
                if batch_data is not None:
                    if total_images > 1:# Si es solo mas de una imagen, emitir el resultado
                         all_batch_data.extend(batch_data)  # Acumular datos válidos
                if result is not None:
                    if total_images > 1:  # Solo guardar en batch si hay más de una imagen
                        all_batch_data.extend(batch_data)
                    self.image_processed.emit(result, valid_count)


                processed_images += 1
                avg_time_per_image = elapsed / (idx + 1) if idx > 0 else 0
                remaining_time = avg_time_per_image * (total_images - (idx + 1))
                progress = int(((idx + 1) / total_images) * 100)  # Fórmula corregida
                elapsed = time.time() - self.start_time
                self.progress_updated.emit(progress, elapsed, remaining_time)
                
           
            self.progress_updated.emit(100, elapsed, 0)    
            self.batch_finished.emit(pd.DataFrame(all_batch_data))

        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")

    def stop(self):
        """ Detener el procesamiento """
        self.is_running = False
    #The process for the images
    def process_single_image(self, image_path, params):
        try:
            start_time = time.time()
            
            # Etapa 1: Cargar imagen
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None, None, None, None
        
            
            clahe = cv2.createCLAHE(clipLimit=params['clip_limit'], tileGridSize=(params['grid_size'], params['grid_size']))
            image = clahe.apply(image) if len(image.shape) == 2 else clahe.apply(image[:, :, 0])
            self.progress_updated.emit(10, start_time, 0)  

            # Etapa 2: Suavizar
            blur = cv2.GaussianBlur(image, (params['blur_size'], params['blur_size']), 0)
           
           

            # Etapa 3: Umbral Otsu
            _, binary = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if params['box'] == True: 
                binary = cv2.bitwise_not(binary)
         
            self.progress_updated.emit(35, start_time, 0) 

            # Etapa 4: Morfología
            kernel_size = params.get('kernel_size', 3)
            morph_iterations = params.get('morph_iterations', 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            binary_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
          
           

            # Etapa 5: Transformada distancia
            dist_transform = cv2.distanceTransform(binary_opened, cv2.DIST_L2, 5)
          
            

            # Etapa 6: Picos locales
            min_distance_peak = params.get('min_distance_peak', 8)
            coordinates = peak_local_max(dist_transform, min_distance=min_distance_peak, threshold_abs=0.5)
            local_max = np.zeros(dist_transform.shape, dtype=bool)
            local_max[tuple(coordinates.T)] = True
         
            

            # Etapa 7: Etiquetado
            markers, _ = ndimage.label(local_max)
        
            self.progress_updated.emit(65, start_time, 0) 
            

            # Etapa 8: Watershed
            labels = watershed(-dist_transform, markers, mask=binary_opened)
          
          

            # Etapa 9: Cálculo porcentaje
            total_pixels = binary_opened.size
            painted_pixels = np.count_nonzero(binary_opened)
            unpainted_percentage = ((total_pixels - painted_pixels) / total_pixels) * 100
            unpainted_percentage_adjusted = max(0, unpainted_percentage - 15.35)
            
            self.progress_updated.emit(75, start_time, 0) 
            

            # Etapa 10: Filtro preliminar
            valid_count = 0
            self.progress_updated.emit(90, start_time, 0) 
        

            # Etapa 11: Procesamiento final
            colored_result, batch_data, valid_count = self.process_bubbles(
                labels, coordinates, params, unpainted_percentage_adjusted, image_path,1
            )
       
            return colored_result, valid_count, unpainted_percentage_adjusted, batch_data

        except Exception as e:
            self.error_occurred.emit(f"Error procesando imagen: {str(e)}")
            return None, None, None, None

    def process_batch(self, image_path, params):
        try:
            start_time = time.time()
            total_stages = 11  # Total de etapas definidas
            current_stage = 0
            
            # Etapa 1: Cargar imagen
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None, None, None, None
            current_stage += 1
            
            clahe = cv2.createCLAHE(clipLimit=params['clip_limit'], tileGridSize=(params['grid_size'], params['grid_size']))
            image = clahe.apply(image) if len(image.shape) == 2 else clahe.apply(image[:, :, 0])


            # Etapa 2: Suavizar
            blur = cv2.GaussianBlur(image, (params['blur_size'], params['blur_size']), 0)
            current_stage += 1
           

            # Etapa 3: Umbral Otsu
            _, binary = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if params['box'] == True:
                binary = cv2.bitwise_not(binary)
            current_stage += 1
            

            # Etapa 4: Morfología
            kernel_size = params.get('kernel_size', 3)
            morph_iterations = params.get('morph_iterations', 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            binary_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
            current_stage += 1
           

            # Etapa 5: Transformada distancia
            dist_transform = cv2.distanceTransform(binary_opened, cv2.DIST_L2, 5)
            current_stage += 1
            

            # Etapa 6: Picos locales
            min_distance_peak = params.get('min_distance_peak', 8)
            coordinates = peak_local_max(dist_transform, min_distance=min_distance_peak, threshold_abs=0.5)
            local_max = np.zeros(dist_transform.shape, dtype=bool)
            local_max[tuple(coordinates.T)] = True
            current_stage += 1
            

            # Etapa 7: Etiquetado
            markers, _ = ndimage.label(local_max)
            current_stage += 1
            

            # Etapa 8: Watershed
            labels = watershed(-dist_transform, markers, mask=binary_opened)
            current_stage += 1
          

            # Etapa 9: Cálculo porcentaje
            total_pixels = binary_opened.size
            painted_pixels = np.count_nonzero(binary_opened)
            unpainted_percentage = ((total_pixels - painted_pixels) / total_pixels) * 100
            unpainted_percentage_adjusted = max(0, unpainted_percentage - 15.35)
            current_stage += 1
          
            

            # Etapa 10: Filtro preliminar
            valid_count = 0
            for lbl in range(1, labels.max() + 1):
                mask = (labels == lbl).astype(np.uint8)
                area_px = cv2.countNonZero(mask)
                x, y, w, h = cv2.boundingRect(mask)
                aspect_ratio = w / h if h > 0 else 0
                if 0.79 <= aspect_ratio <= 2.5 and params['min_area_threshold'] <= area_px <= params['max_area_threshold']:
                   pass
            current_stage += 1
        

            # Etapa 11: Procesamiento final
            colored_result, batch_data, valid_count = self.process_bubbles(
                labels, coordinates, params, unpainted_percentage_adjusted, image_path,2
            )
       
            return colored_result, valid_count, unpainted_percentage_adjusted, batch_data

        except Exception as e:
            self.error_occurred.emit(f"Error procesando imagen: {str(e)}")
            return None, None, None, None



    def process_bubbles(self, labels, coordinates, params, unpainted_percentage, image_path, type):
        colored_result = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        batch_data = []
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        escala_mm_px = 4.5 / 208
        valid_count = 0  # Inicializar contador



        for lbl in range(1, labels.max() + 1):
                mask = (labels == lbl).astype(np.uint8)
                area_px = cv2.countNonZero(mask)
                x, y, w, h = cv2.boundingRect(mask)
                aspect_ratio = w / h if h > 0 else 0

                if 0.7 <= aspect_ratio < 2.0 and params['min_area_threshold'] <= area_px <= params['max_area_threshold'] and type == 2:
                    # Cálculos de métricas
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    perimeter_px = cv2.arcLength(contours[0], True) if contours else 0.0
                    diametro_px = math.sqrt((4 * area_px) / math.pi)
                    class_id = self.classify_by_size(area_px, increase=150)  
                    color = self.get_color_by_increment(area_px, 150)
                    id_value = str(valid_count)
                    # Añadir a resultados
                    batch_data.append({
                        'Imagen_idobject': f"{image_name}_{len(batch_data)}",
                        'Area_px': area_px,
                        'Area_mm2': area_px * (escala_mm_px ** 2),
                        'Perimetro_mm': perimeter_px * escala_mm_px,
                        'Diametro_mm': diametro_px * escala_mm_px,
                        'Porcentaje_no_pintado': unpainted_percentage,
                    # 'Coordenadas': bubble_coordinates.tolist(), Add a button futhermore
                        'aspect_ratio': aspect_ratio,
                        'Class_id' : class_id
                    })
                    valid_count += 1
                    colored_result[labels == lbl] = color   
                    
                        #Funciones unicas no se mueven
                
       

        return colored_result, batch_data, valid_count
   
    def get_color_by_increment(self,area, increment=150):
                # Determine the class (range) of the area
                class_id = area // increment  # Integer division
                random.seed(class_id)  # Ensure consistent color for the same class
                return (
                    random.randint(0, 255), 
                    random.randint(0, 255), 
                    random.randint(0, 255)
                ) 
    def classify_by_size(self, area, increase=150):
        """Clasifica las burbujas por tamaño."""
        class_id = area // increase
        return class_id


# Clean the image by putting white figures 
class CleaningProcessor(QThread): 
    progress_updated = pyqtSignal(int)      # Progreso del procesamiento
    image_processed = pyqtSignal(str)         # Ruta de la imagen procesada
    path_processed  = pyqtSignal(str)         # Ruta general del directorio temporal
    error_occurred = pyqtSignal(str)          # Señal de error
    processing_finished = pyqtSignal(bool)    # Señal de finalización del procesamiento
    
    def __init__(self, image_paths, custom_coords):
        super().__init__()
        self.image_paths = image_paths
        self.is_running = True
        self.custom_coords = custom_coords  # Por ejemplo, self.custom_selections
        self.avg_time_per_image = 0
        self.temp_dir = tempfile.mkdtemp()
    
    def run(self):
        try:
            self.start_time = time.time()
            total_images = len(self.image_paths)
            
            # Rectángulos por defecto en caso de que una imagen no tenga selecciones personalizadas
            default_rects = [
                [(589, 860), (1002, 858), (590, 934), (1002, 933)],
                [(480, 114), (481, 2), (1, 2), (1, 114)],
                [(1178, 2), (1177, 251), (1470, 252), (1470, 2)],
                [(190, 724), (1, 723), (4, 934), (191, 931)],
                [(1325, 861), (1468, 864), (1468, 931), (1327, 933)]
            ]
            
            for i, img_path in enumerate(self.image_paths):
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                # Convertir a RGB (si es necesario) para trabajar con 3 canales
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                # Si hay selecciones personalizadas para esta imagen, aplicarlas todas
                if self.custom_coords is not None and img_path in self.custom_coords:
                    pts_list = []
                    for polygon in self.custom_coords[img_path]:
                        pts = np.array(polygon, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        pts_list.append(pts)
                    # Una única llamada a fillPoly para rellenar todas las áreas
                    cv2.fillPoly(image, pts_list, (255, 255, 255))
                else:
                    # Si no hay selecciones personalizadas, dibujar los rectángulos por defecto
                    for rect_points in default_rects:
                        image = self.draw_rectangle_from_points(image, rect_points)
                
                # Guardar la imagen procesada en la carpeta temporal
                filename = os.path.basename(img_path)
                temp_output_path = os.path.join(self.temp_dir, filename)
                cv2.imwrite(temp_output_path, image)
                
                # Emitir la señal con la ruta de la imagen procesada
                self.image_processed.emit(temp_output_path)
                
                # Actualizar el progreso
                progress = int((i + 1) / total_images * 100)
                self.progress_updated.emit(progress)
            
            # Emitir la señal con la ruta general de la carpeta temporal
            self.path_processed.emit(self.temp_dir)
        
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")
    
    def draw_rectangle_from_points(self, image, points, color=(255, 255, 255), thickness=-1):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
    
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
    
        top_left = (min_x, min_y)
        bottom_right = (max_x, max_y)
    
        cv2.rectangle(image, top_left, bottom_right, color, thickness)
    
        return image

class Tranform_files(QThread):
    path_merged = pyqtSignal(str)
    path_converted = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    error_signal = pyqtSignal(str, str)
    success_signal = pyqtSignal(str, int, int)


    def __init__(self, path_origin, type_process,start_number, increment):
            super().__init__()
            self.path_origin = path_origin
            self.start_number = start_number
            self.increment = increment
            self.type_process = type_process
            self.temp_dir = tempfile.mkdtemp()
            
            if self.type_process == 1:
                if isinstance(self.path_origin, list):
                    self.files = path_origin
                else:
                    # Si se pasa un solo archivo, lo ponemos en una lista
                    self.files = [path_origin]
            else: #No aplicado otro tipo de proceso
             
                self.files = None  
                
    def run(self):
        try:
            if self.type_process == 1:
                print(">> Ejecutando merge_files...")
                self.merge_files()
             
            if self.type_process == 2:
               self.convert_files(self.path_origin,self.start_number, self.increment)
                
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")
    """Combina archivos manteniendo cada uno en tablas separadas"""
    
    def merge_files(self):
        try:
            print("Working...")
            if not self.files:
                self.error_signal.emit("Error", "No se seleccionaron archivos")
                return

            if not self.validate_file_types(self.files):
                self.error_signal.emit("Error", "Todos los archivos deben ser del mismo tipo")
                return

            print("Working")
            merged_df, total_files, total_records = self.process_files()
            output_path = self.save_file(merged_df)
            if output_path:
                self.success_signal.emit(output_path, total_files, total_records)
        except Exception as e:
            self.error_signal.emit("Error crítico", f"Error durante el merge: {str(e)}")

    def process_files(self):
        # Tu lógica original de lectura y validación de archivos
        dfs = []
        reference_columns = None
        file_type = os.path.splitext(self.files[0])[1].lower()

        for index, file_path in enumerate(self.files):
            print("Transforming....", file_path)
            try:
                # Leer el archivo según su tipo
                if file_type == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_csv(file_path, delimiter='\t')

                # Agregar una columna que indique el archivo de origen para cada fila
                df['source_file'] = os.path.basename(file_path)

                # Validar que todas las columnas sean iguales en cada archivo
                if reference_columns is None:
                    reference_columns = list(df.columns)
                else:
                    if list(df.columns) != reference_columns:
                        self.error_signal.emit(
                            "Error de columnas",
                            f"Archivo {os.path.basename(file_path)} tiene columnas diferentes"
                        )
                        return None, 0, 0

                dfs.append(df)
            except Exception as e:
                self.error_signal.emit(
                    "Error de lectura",
                    f"Error leyendo {os.path.basename(file_path)}:\n{str(e)}"
                )
                return None, 0, 0

        concatenated = pd.concat(dfs, ignore_index=True)
        return concatenated, len(dfs), len(concatenated)


    def save_file(self, merged_df):
        # Guardar el archivo combinado
            file_type = os.path.splitext(self.files[0])[1].lower()
            output_filename = f"merged_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}{file_type}"
            
            output_path = os.path.join(
                os.path.dirname(self.files[0]),  # Directorio del primer archivo
                output_filename
            )
        
            try:
                if file_type == '.csv':
                    merged_df.to_csv(output_path, index=False)
                else:
                    merged_df.to_csv(output_path, sep='\t', index=False)
                return output_path
            except Exception as e:
                self.error_signal.emit("Error al guardar", str(e))
                return None

    def validate_file_types(self, file_paths):
        # Tu método de validación original
        extensions = [os.path.splitext(f)[1].lower() for f in file_paths]
        return all(ext in ['.csv', '.txt'] for ext in extensions) and len(set(extensions)) == 1




    def convert_files(self, csv_filename,start_number, increment):
        try:
            print("🟡 Iniciando conversión de archivos...")
            df = pd.read_csv(csv_filename)
            print("✅ Archivo CSV leído correctamente.")

            # Paso 1: Validar y procesar columnas clave
            if 'Imagen_idobject' not in df.columns:
                raise ValueError("❌ Error: La columna 'Imagen_idobject' no existe.")

            # Extraer 'Coordenate_type' (primeros 2 caracteres)
            df['Coordenate_type'] = df['Imagen_idobject'].str[0:2]
            
            # Extraer 'Imagen_idobject_substr' (caracteres 2-5, sin guiones)
            df['Imagen_idobject_substr'] = (
                df['Imagen_idobject']
                .str[2:5]
                .str.replace('_', '', regex=True)
                .pipe(pd.to_numeric, errors='coerce')
            )
            
            # Eliminar filas inválidas
            df = df.dropna(subset=['Imagen_idobject_substr'])
            df['Imagen_idobject_substr'] = df['Imagen_idobject_substr'].astype(int)
            min_value = df['Imagen_idobject_substr'].min()
            df['Imagen_idobject_substr'] = df['Imagen_idobject_substr'] - min_value

            # Paso 2: Agrupar y fusionar incluyendo SIEMPRE 'Coordenate_type'
            # Crear tabla base con conteos
            count_table = (
                df.groupby(['Coordenate_type', 'Imagen_idobject_substr'])
                .size()
                .reset_index(name='Count')
            )

            # Fusionar promedio de 'Porcentaje_no_pintado'
            avg_porcentaje = (
                df.groupby(['Coordenate_type', 'Imagen_idobject_substr'])
                ['Porcentaje_no_pintado'].mean()
                .reset_index()
            )
            count_table = pd.merge(count_table, avg_porcentaje, on=['Coordenate_type', 'Imagen_idobject_substr'])

            # Fusionar promedio de 'Diametro_mm'
            avg_diametro = (
                df.groupby(['Coordenate_type', 'Imagen_idobject_substr'])
                ['Diametro_mm'].mean()
                .reset_index()
            )
            count_table = pd.merge(count_table, avg_diametro, on=['Coordenate_type', 'Imagen_idobject_substr'])

            # Fusionar moda de 'Class_id'
            mode_class = (
                df.groupby(['Coordenate_type', 'Imagen_idobject_substr'])
                ['Class_id'].agg(lambda x: x.mode()[0])
                .reset_index()
            )
            count_table = pd.merge(count_table, mode_class, on=['Coordenate_type', 'Imagen_idobject_substr'])

            # Paso 3: Calcular 'Scene_coordinate_system' (sin bucles)
            count_table['Scene_coordinate_system'] = (
                #-21.00000 + 0.5 * (count_table['Imagen_idobject_substr'] - 78)
                start_number + increment * (count_table['Imagen_idobject_substr'])
            )

            # Paso 4: Ordenar y guardar
            count_table = count_table.sort_values(by=['Coordenate_type', 'Imagen_idobject_substr'])
            
            count_table = count_table.rename(columns={
                'Porcentaje_no_pintado': 'Avg_Porcentaje_no_pintado',
                'Diametro_mm': 'Avg_Diametro_mm',
                'Class_id': 'Mode_Class_id'
            })

            output_filename = os.path.join(self.temp_dir, f"converted_{os.path.basename(csv_filename)}")
            count_table['Count'] = count_table['Count']/2
            count_table.to_csv(output_filename, index=False)
            print(f"✅ Archivo convertido guardado en: {output_filename}")

            self.path_converted.emit(output_filename)
            return output_filename

        except Exception as e:
            print(f"❌ Error crítico: {str(e)}")
            raise

#-------------------QWIDGET DE CADA PAGINA------------------------
class CleanPage(QWidget):
    def __init__(self):
        super().__init__()
        self.custom_selections = {}  # Aquí se almacenarán las selecciones (por imagen)
        self.init_ui()
        
    def init_ui(self):
        self.image_paths = []
        self.current_image_path = None
        self.batch_data = []
        self.params = {}
        self.result_image = None
        self.processor = None
        self.start_time = None
        self.timer = QTimer(self)
        self.control_layout = QVBoxLayout()
        self.timer.timeout.connect(self.update_elapsed_time)
        self.setStyleSheet("background-color: ##e8f8f5;")

        # Panel izquierdo con controles
        self.control_widget = QWidget()
        self.control_widget.setLayout(self.control_layout)

        self.image_widget = QLabel()
        self.image_widget.setFixedSize(800, 800)
        self.image_widget.setAlignment(Qt.AlignCenter)
        self.image_widget.setText("Limpieza de imagenes.")
        self.image_widget.setStyleSheet("QLabel { color : gray; font: 18px; }")

        # Botón para seleccionar directorio de imágenes
        self.select_dir_button = QPushButton("Seleccionar Carpeta")
        self.select_dir_button.setStyleSheet("""
            QPushButton {
                background-color: #c8c241; /* Verde */
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.select_dir_button.clicked.connect(self.select_directory)
        self.control_layout.addWidget(self.select_dir_button)


        # Lista de imágenes
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.image_selected)
        self.control_layout.addWidget(self.image_list)

        # Botón para activar la selección de lazo
        self.start_selection_button = QPushButton("Activar selección de lazo")
        self.start_selection_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; /* Verde */
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #337436;
            }
        """)
        self.start_selection_button.clicked.connect(self.start_selection)
        self.control_layout.addWidget(self.start_selection_button)

        # Botón para procesar todas las imágenes
        self.process_all_button = QPushButton("Procesar lazos creados")
        self.process_all_button.setStyleSheet("""
            QPushButton {
            background-color: #2196F3; /* Azul */
            color: white;
            font-weight: bold;
            font-size: 14px;
            padding: 10px;
            border-radius: 5px;
            }
            QPushButton:hover {
            background-color: #1d79ca;
            }
        """)
        self.process_all_button.clicked.connect(self.convert_images)
        self.control_layout.addWidget(self.process_all_button)

        # Botón para eliminar el último lazo
        self.erased_the_last = QPushButton("Eliminar último lazo")
        self.erased_the_last.setStyleSheet("""
            QPushButton {
                background-color: #FF9800; /* Naranja */ 
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #831919;
            }
        """)
        self.erased_the_last.clicked.connect(self.erased_selection)
        self.control_layout.addWidget(self.erased_the_last)

        # Botón para limpiar los lazos creados en la imagen actual
        self.clear_sel_button = QPushButton("Eliminar todos los lazos creados")
        self.clear_sel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336; /* Rojo */
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #620e0e;
            }
        """)
        self.clear_sel_button.clicked.connect(self.clear_selection)
        self.control_layout.addWidget(self.clear_sel_button)

        # Botón para guardar los resultados
        self.save_button = QPushButton("Exportar Resultados")
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0; /* Púrpura */
                color: white;
                font-weight: bold;
                font-size: 14px;                                      
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #75208b;
            }
        """)
        self.control_layout.addWidget(self.save_button)
            

        

       
        
        self.save_button.clicked.connect(self.save_image)
        self.control_layout.addWidget(self.save_button)  # 🔹 Agregar a la UI

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.control_layout.addWidget(self.progress_bar)

        self.time_label = QLabel()
        self.time_label.setText("Tiempo restante: N/A")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("QLabel { color : black; font: 16px; }")
        self.control_layout.addWidget(self.time_label)


           # Panel de imagen usando LassoLabel para selección interactiva
        self.image_widget = LassoLabel()
        self.image_widget.setFixedSize(800, 800)
        self.image_widget.setAlignment(Qt.AlignCenter)
        self.image_widget.setText(
        "<div style='font-size:21px;'>"
        "Carga una imagen y presiona 'Activar lazo' para dibujar.<br>"
        "Usa clic izquierdo para dibujar; al soltar se cierra y se llena el lazo.<br>"
        "Si una imagen no es aplicado  algun lazo, se  aplicara la limpieza predeterminada."
        "</div>"
                                )
        # Connect the signal so that when a polygon is finished its coordinates are saved.
        self.image_widget.selectionFinished.connect(self.save_custom_selection)

        



        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.control_widget)
        self.splitter.addWidget(self.image_widget)

       
        # 🔹 Crear el layout principal
        self.main_layout = QHBoxLayout(self)
        self.main_layout.addWidget(self.splitter)
        self.setLayout(self.main_layout)  # Asignar el layout principal a la ventana

    def start_selection(self):
        # Enable the selection mode so that LassoLabel starts drawing.
        self.image_widget.enableSelection()

    def select_directory(self):
        """Permite al usuario seleccionar un directorio de imágenes."""
        dir_path = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio de Imágenes", "")

        if dir_path:
            # 🔹 Si el procesamiento ha terminado, eliminar carpeta temporal
            if hasattr(self.processor, 'processing_finished') and self.processor.processing_finished:
                shutil.rmtree(self.processor.temp_dir, ignore_errors=True)
                print("✅ Directorio temporal eliminado.")

            # 🔹 Limpiar lista y variables antes de actualizar
            self.image_paths = []
            self.image_list.clear()
            self.current_image_path = None  # Evita seleccionar una imagen anterior

            # Cargar imágenes del nuevo directorio
            self.image_paths = [
                os.path.join(dir_path, f) for f in os.listdir(dir_path)
                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))
            ]
            
            # Mostrar en la UI
            for img_path in self.image_paths:
                self.image_list.addItem(os.path.basename(img_path))

            # Mensaje informativo
            if not self.image_paths:
                QMessageBox.warning(self, "Advertencia", "No se encontraron imágenes en el directorio seleccionado.")
            else:
                QMessageBox.information(self, "Información", f"Se encontraron {len(self.image_paths)} imágenes.")

            # 🔹 Actualizar la vista para que no muestre imágenes temporales
            self.display_result(None)  # Borra la imagen mostrada

    def image_selected(self, item):
        """Seleccionar imagen de la lista y mostrarla en la interfaz."""
        image_name = item.text()
        
        # Buscar en las imágenes originales
        for path in self.image_paths:
            if os.path.basename(path) == image_name:
                self.current_image_path = path
                break
        self.image_widget.clearPolygons()
        self.display_image(self.current_image_path)

        # 🔹 Verificar si hay una versión procesada en el directorio temporal
        if self.processor is not None:
            temp_image_path = os.path.join(self.processor.temp_dir, image_name)
            if os.path.exists(temp_image_path):
                self.current_image_path = temp_image_path  # Usa la versión procesada
        
        self.display_image(self.current_image_path)

    def convert_images(self):
        """ Procesar todas las imágenes en lote """
        self.start_time = time.time() 
        if not self.image_paths:
            QMessageBox.warning(self, "Error", "No hay imágenes cargadas.")
            return
        
        # Detener cualquier procesamiento previo
        if self.processor and self.processor.isRunning():
            self.processor.stop()

        # Mostrar "Cargando" en el QLabel
        self.image_widget.setText("Procesando lote...")
        self.image_widget.setStyleSheet("QLabel { color : black; font: 18px; }")
        
        custom_coords_dict = {}
        if hasattr(self, 'custom_selections') and self.custom_selections:
            for img_path, polygons in self.custom_selections.items():
                # Se asume que polygons es una lista de selecciones y se toma la última
                custom_coords_dict[img_path] = polygons
            
       
            
        # Configurar y lanzar el hilo
        self.processor = CleaningProcessor(self.image_paths, custom_coords_dict)
        self.processor.progress_updated.connect(self.progress_bar.setValue)
        self.processor.progress_updated.connect(self.update_progress)
        self.processor.error_occurred.connect(self.show_error)

        #self.processor.processing_finished.connect(self.on_image_processed)
        self.processor.image_processed.connect(self.on_image_processed)
        self.processor.path_processed.connect(self.on_path_processed)  # 🔹 Conectar señal
       
       
        self.save_button = QPushButton("Guardar Imagen", self)
        self.processor.start()
        QApplication.processEvents()  # Actualizar la interfaz
        #Display results image 

    def on_image_processed(self, temp_image_path):
        """Actualizar la GUI con la imagen procesada."""
        self.display_result(temp_image_path)
        self.save_button.setEnabled(True)  # 🔹 Activar botón de guardado
        QApplication.processEvents()  # Forzar actualización de la interfaz

    def display_result(self, image_path):
        """Muestra la imagen procesada en el QLabel."""
        if image_path is None or not os.path.exists(image_path):
            self.image_widget.setText("No se pudo procesar la imagen.")
            self.image_widget.setStyleSheet("QLabel { color : red; font: 15px; }")
            return

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("No se pudo cargar la imagen.")

            # Convertir a formato RGB si es necesario
            if len(image.shape) == 2:  
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Redimensionar y mostrar en QLabel
            height, width = image.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                self.image_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_widget.setPixmap(scaled_pixmap)

        except Exception as e:
            self.image_widget.setText(f"Error al mostrar la imagen: {str(e)}")
            self.image_widget.setStyleSheet("QLabel { color : red; font: 15px; }")

    def on_path_processed(self, temp_dir):
        """Actualizar la ruta procesada y habilitar el botón de guardado."""
        print("Activando botón de guardado para:", temp_dir)
        self.temp_dir = temp_dir
        self.save_button.setEnabled(True)

    def save_image(self):
        """Permitir al usuario guardar las imágenes procesadas en una ubicación deseada."""
        if not hasattr(self, 'temp_dir') or not self.temp_dir:
            QMessageBox.warning(self, "Advertencia", "No hay imágenes procesadas para guardar.")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Guardar imágenes en", "")
        if save_dir:
            for filename in os.listdir(self.temp_dir):
                temp_image_path = os.path.join(self.temp_dir, filename)
                save_path = os.path.join(save_dir, filename)
                shutil.move(temp_image_path, save_path)  # 🚀 Mover la imagen
            QMessageBox.information(self, "Éxito", "Imágenes guardadas correctamente.")
    def show_error(self, message):
        """ Mostrar mensajes de error """
        QMessageBox.critical(self, "Error", message)
    def update_progress(self, progress):
        elapsed = time.time() - self.start_time
        if progress > 0:
            total_time = elapsed / (progress / 100)
            remaining = total_time - elapsed
        else:
            remaining = 0

        # Formatear tiempo
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining))

        # Actualizar barra de progreso
        self.progress_bar.setValue(progress)
        print(f"Progreso: {progress}% | Tiempo transcurrido: {elapsed_str} | Restante: {remaining_str}")

        # Actualizar label según el mod o
        if len(self.image_paths) > 1:
            # Modo de procesamiento en lote
            text = f"Procesando lote... {progress}%\nTiempo transcurrido: {elapsed_str}\nRestante: {remaining_str}"
        else:
            # Modo de procesamiento de una sola imagen
            text = f"Procesando imagen... {progress}%\nTiempo transcurrido: {elapsed_str}\nRestante: {remaining_str}"

        self.time_label.setText(text)
        
        QApplication.processEvents()  # Forzar actualización de la interfaz
    def update_elapsed_time(self):
        """Actualiza el tiempo transcurrido cada segundo en la GUI."""
        if self.start_time is None:
            return  # Evita errores si el tiempo de inicio aún no está definido
        
        elapsed = time.time() - self.start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    def display_image(self, image_path):
        image = cv2.imread(image_path)
        if image is not None:
            # Guardamos las dimensiones originales para el mapeo
            self.current_image = QImage(image.data, image.shape[1], image.shape[0],
                                        3 * image.shape[1], QImage.Format_RGB888)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
          
            q_image = QImage(image.data, width, height,  3 * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_widget.original_image_size = pixmap.size()
            self.image_widget.setAlignment(Qt.AlignCenter)
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(self.image_widget.size(), Qt.KeepAspectRatio)
            self.image_widget.setPixmap(scaled_pixmap)
            # Aquí actualizamos _pixmapRect: se asume que el QLabel centra y escala la imagen.
            # Puedes calcularlo a partir de self.image_widget.size() y el tamaño del pixmap.
            label_size = self.image_widget.size()
            pixmap_size = scaled_pixmap.size()
            x = (label_size.width() - pixmap_size.width()) // 2
            y = (label_size.height() - pixmap_size.height()) // 2
            self._pixmapRect = QRect(x, y, pixmap_size.width(), pixmap_size.height())
        else:
            self.image_widget.setText("Imagen no válida o no se pudo cargar.")

    
    def map_widget_to_image(self, widget_x, widget_y):
        """ Convierte coordenadas del QLabel (widget) a la imagen original """
        if not self._pixmapRect or not hasattr(self, 'current_image'):
            return widget_x, widget_y  # No hay imagen cargada, devolver original

        # 🔹 Factor de escala único basado en la relación ancho
        scale_factor = self.current_image.width() / self._pixmapRect.width()

        # 🔹 Ajustar coordenadas considerando desplazamientos (offsets)
        image_x = (widget_x) * scale_factor
        image_y = (widget_y ) * scale_factor


        return int(image_x), int(image_y)




    def clear_selection(self):
        """Limpia los lazos (polígonos) dibujados en el widget de la imagen actual."""
        self.image_widget.clearPolygons()
        QMessageBox.information(self, "Selección Eliminada", "Se han eliminado los lazos de la imagen actual.")
    def save_custom_selection(self, polygon):
            # Suponiendo que 'polygon' es una lista de QPoint obtenidos desde el widget.
            polygon_coords = [self.map_widget_to_image(pt[0], pt[1]) for pt in polygon]
          

            if not self.current_image_path:
                return
            if not hasattr(self, 'polygon_data'):
                self.polygon_data = []


            if self.current_image_path not in self.custom_selections:
                self.custom_selections[self.current_image_path] = []
            self.custom_selections[self.current_image_path].append(polygon_coords)

            self.polygon_data.append((self.current_image_path, polygon_coords))

            QMessageBox.information(
                self,
                "Selección Guardada",
                f"Se guardó la selección para {os.path.basename(self.current_image_path)}.\n"
                f"Número total de lazos: {len(self.custom_selections[self.current_image_path])}"
            )

            print("Datos de polígonos guardados:", self.polygon_data)

    def erased_selection(self):
        if hasattr(self, 'polygon_data') and self.polygon_data:
            removed_polygon = self.polygon_data.pop()
            self.image_widget.polygons.pop()
            self.image_widget.update()
            print(f"Removed polygon: {removed_polygon}")
        else:
            print("No polygons to remove.")


class ProcessPage(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.image_paths = []
        self.current_image_path = None
        self.batch_data = []
        self.params = {}
        self.result_image = None
        self.processor = None
        self.start_time = None
        self.timer = QTimer(self)
        self.control_layout = QVBoxLayout()
        self.timer.timeout.connect(self.update_elapsed_time)
        self.setStyleSheet("background-color: #e8f8f5;")

        # Panel izquierdo con controles
        self.control_widget = QWidget()
        self.control_widget.setLayout(self.control_layout)

        self.image_widget = QLabel()
        self.image_widget.setFixedSize(800, 800)
        self.image_widget.setAlignment(Qt.AlignCenter)
        self.image_widget.setText("Detector de Burbujas fase Beta. Seleccione un directorio de imágenes para comenzar.")
        self.image_widget.setStyleSheet("QLabel { color : gray; font: 18px; }")

        # Botón para seleccionar directorio de imágenes
        self.select_dir_button = QPushButton("Seleccionar Directorio de Imágenes")
        self.select_dir_button.setStyleSheet("""
            QPushButton {
                background-color: #c8c241; /* Verde */
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.select_dir_button.clicked.connect(self.select_directory)
        self.control_layout.addWidget(self.select_dir_button)

        # Lista de imágenes
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.image_selected)
        self.control_layout.addWidget(self.image_list)

        # Controles de parámetros
        self.parameters_layout = QFormLayout()
        # Configuración de parámetros para el procesamiento de imágenes

        self.box_selection_active = False
        
        self.box_selection_button = QPushButton("Desactivado")
        self.box_selection_button.setStyleSheet("""
            QPushButton {
            background-color: ##A71A1B;
            color: white;
            font-weight: bold;
            padding: 6px;
            border-radius: 3px;
            }
            QPushButton:hover {
            background-color: #7D1112;
            }
        """)
        self.box_selection_button.clicked.connect(lambda: [
            setattr(self, 'box_selection_active', not self.box_selection_active),
            self.box_selection_button.setText("Activado" if self.box_selection_active else "Desactivado"),
            self.box_selection_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 6px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            """ if self.box_selection_active else
            """
            QPushButton {
                background-color: #A71A1B;
                color: white;
                font-weight: bold;
                padding: 6px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #7D1112;
            }
            """
            )
        ])
        self.parameters_layout.addRow("Invertir Deteccion:", self.box_selection_button)

        # Límite de recorte para CLAHE
        self.clip_limit_spin = QSpinBox()
        self.clip_limit_spin.setRange(1, 10)
        self.clip_limit_spin.setValue(6)
        self.parameters_layout.addRow("Límite de recorte CLAHE:", self.clip_limit_spin)

        # Tamaño de la cuadrícula para CLAHE
        self.grid_size_spin = QSpinBox()
        self.grid_size_spin.setRange(1, 16)
        self.grid_size_spin.setValue(16)
        self.parameters_layout.addRow("Tamaño de cuadrícula CLAHE:", self.grid_size_spin)

        # Tamaño de suavizado de imagen
        self.blur_size_spin = QSpinBox()
        self.blur_size_spin.setRange(1, 15)
        self.blur_size_spin.setValue(5)
        self.blur_size_spin.setSingleStep(2)
        self.parameters_layout.addRow("Tamaño de suavizado:", self.blur_size_spin)

        # Tamaño del kernel para el filtro de detección
        self.kernel_size_spin = QSpinBox()
        self.kernel_size_spin.setRange(1, 15)
        self.kernel_size_spin.setValue(3)
        self.kernel_size_spin.setSingleStep(2)
        self.parameters_layout.addRow("Tamaño del kernel de detección:", self.kernel_size_spin)

        # Número de iteraciones para operaciones morfológicas
        self.morph_iterations_spin = QSpinBox()
        self.morph_iterations_spin.setRange(1, 10)
        self.morph_iterations_spin.setValue(2)
        self.parameters_layout.addRow("Número de iteraciones morfológicas:", self.morph_iterations_spin)

        # Distancia mínima entre picos detectados
        self.min_distance_peak_spin = QSpinBox()
        self.min_distance_peak_spin.setRange(1, 20)
        self.min_distance_peak_spin.setValue(8)
        self.parameters_layout.addRow("Distancia mínima entre picos:", self.min_distance_peak_spin)

        # Aplicar estilos a los QSpinBox
        spinbox_style = """
            QSpinBox {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
                font-size: 14px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 16px;
                height: 16px;
            }
        """

        self.clip_limit_spin.setStyleSheet(spinbox_style)
        self.grid_size_spin.setStyleSheet(spinbox_style)
        self.blur_size_spin.setStyleSheet(spinbox_style)
        self.kernel_size_spin.setStyleSheet(spinbox_style)
        self.morph_iterations_spin.setStyleSheet(spinbox_style)
        self.min_distance_peak_spin.setStyleSheet(spinbox_style)

        # Agregar los controles de parámetros al layout principal
        self.control_layout.addLayout(self.parameters_layout)

        # Botón para procesar una imagen individual
        self.process_image_button = QPushButton("Procesar Imagen")
        self.process_image_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.process_image_button.clicked.connect(self.process_image)
        self.control_layout.addWidget(self.process_image_button)

        # Botón para procesar todas las imágenes
        self.process_all_button = QPushButton("Procesar Todas las Imágenes")
        self.process_all_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1e88e5;
            }
        """)
        self.process_all_button.clicked.connect(self.process_all_images)
        self.control_layout.addWidget(self.process_all_button)

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.control_layout.addWidget(self.progress_bar)

        self.time_label = QLabel()
        self.time_label.setText("Tiempo restante: N/A")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("QLabel { color : black; font: 16px; }")
        self.control_layout.addWidget(self.time_label)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.control_widget)
        self.splitter.addWidget(self.image_widget)

        # 🔹 Crear el layout principal
        self.main_layout = QHBoxLayout(self)
        self.main_layout.addWidget(self.splitter)
        self.setLayout(self.main_layout)  # Asignar el layout principal a la ventana

    def graphing(self):
        # Placeholder for graphing functionality
        QMessageBox.information(self, "Graphing", "Graphing functionality is not implemented yet.")
        return
    
    def update_progress(self, progress, elapsed, remaining):
        elapsed = time.time() - self.start_time
        if progress > 0:
            total_time = elapsed / (progress / 100)
            remaining = total_time - elapsed
        else:
            remaining = 0

        # Formatear tiempo
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining))

        # Actualizar barra de progreso
        self.progress_bar.setValue(progress)
        print(f"Progreso: {progress}% | Tiempo transcurrido: {elapsed_str} | Restante: {remaining_str}")

        # Actualizar label según el modo
        if len(self.image_paths) > 1:
            # Modo de procesamiento en lote
            text = f"Procesando lote... {progress}%\nTiempo transcurrido: {elapsed_str}\nRestantes: {remaining_str}"
        else:
            # Modo de procesamiento de una sola imagen
            text = f"Procesando imagen... {progress}%\nTiempo transcurrido: {elapsed_str}\nRestante: {remaining_str}"

        self.time_label.setText(text)
        
        QApplication.processEvents()  # Forzar actualización de la interfaz

    #Función para seleccionar un directorio de imágenes(Do not touch)
    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio de Imágenes", "")
        if dir_path:
            self.image_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                                if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.tif')]
            self.image_list.clear()
            for img_path in self.image_paths:
                self.image_list.addItem(os.path.basename(img_path))
            if not self.image_paths:
                QMessageBox.warning(self, "Advertencia", "No se encontraron imágenes en el directorio seleccionado. Considere que solo se admiten archivos .jpg, .png, .jpeg y .tif.")
            else:
                QMessageBox.information(self, "Información", f"Se encontraron {len(self.image_paths)} imágenes.")

    def image_selected(self, item):
        image_name = item.text()
        for path in self.image_paths:
            if os.path.basename(path) == image_name:
                self.current_image_path = path
                break
        self.display_image(self.current_image_path)

    def display_image(self, image_path):
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(self.image_widget.size(), Qt.KeepAspectRatio)
            self.image_widget.setPixmap(scaled_pixmap)
        else:
            self.image_widget.setText("Imagen no válida o no se pudo cargar.")
            self.image_widget.setStyleSheet("QLabel { color : red; font: 15px; }")

    def get_parameters(self):
        self.params = {
            'max_area_threshold': 15000,
            'min_area_threshold': 0,
            'clip_limit': self.clip_limit_spin.value(),
            'grid_size': self.grid_size_spin.value(),
            'blur_size': self.blur_size_spin.value(),
            'kernel_size': self.kernel_size_spin.value(),
            'morph_iterations': self.morph_iterations_spin.value(),
            'min_distance_peak': self.min_distance_peak_spin.value(),
            # 'box' will be True if the box selection is active, otherwise False.
            'box': True if self.box_selection_active else False
            # Agrega más parámetros si es necesario
        }
    #Error about the image
    def process_image(self):
            """ Procesar una sola imagen en segundo plano """
            self.start_time = time.time()
            if not self.current_image_path:
                QMessageBox.warning(self, "Error", "Selecciona una imagen primero.")
                return
            self.start_time = time.time()
            self.timer.start(1000)
            # Detener cualquier procesamiento previo
            if self.processor and self.processor.isRunning():
                self.processor.stop()
             # Mostrar "Cargando" en el QLabel
            self.image_widget.setText("Cargando...")
            self.image_widget.setStyleSheet("QLabel { color : black; font: 18px; }")        
            # Configurar y lanzar el hilo
            self.get_parameters()
            self.processor = ImageProcessor([self.current_image_path], self.params)
            self.processor.image_processed.connect(self.on_image_processed)
            self.processor.progress_updated.connect(self.progress_bar.setValue)
            self.processor.batch_finished.connect(self.on_batch_finished) 
            #self.processor.progress_updated.connect(self.update_progress)
            self.processor.error_occurred.connect(self.show_error)
            QApplication.processEvents()  # Actualizar la interfaz
            self.processor.start()

    def process_all_images(self):
        """ Procesar todas las imágenes en lote """
        self.start_time = time.time() 
        if not self.image_paths:
            QMessageBox.warning(self, "Error", "No hay imágenes cargadas.")
            return
        
        # Detener cualquier procesamiento previo
        if self.processor and self.processor.isRunning():
            self.processor.stop()

        # Mostrar "Cargando" en el QLabel
        self.image_widget.setText("Procesando lote...")
        self.image_widget.setStyleSheet("QLabel { color : black; font: 18px; }")
        
       
            
        # Configurar y lanzar el hilo
        self.get_parameters()
        self.processor = ImageProcessor(self.image_paths, self.params)
        self.processor.progress_updated.connect(self.progress_bar.setValue)

        self.processor.progress_updated.connect(self.update_progress)
        self.processor.error_occurred.connect(self.show_error)
        self.processor.batch_finished.connect(self.on_batch_finished)
        self.processor.image_processed.connect(self.on_image_processed)
        QApplication.processEvents()  # Actualizar la interfaz
        self.processor.start()
        

    def on_image_processed(self, result_image, valid_count):
        """ Actualizar la GUI con los resultados de una imagen """
        self.display_result(result_image)
        
        print(self, "Éxito", f"Burbujas detectadas: {valid_count}")

    def on_batch_finished(self, results_df):
        self.timer.stop()  # Detener el temporizador cuando el procesamiento finaliza
        if not results_df.empty:
            self.results_df = results_df
            self.download_button = QPushButton(f"Descargar CSV {len(self.control_layout.findChildren(QPushButton)) + 1}", self)
            self.download_button.clicked.connect(lambda: self.download_csv(self.results_df))
            self.control_layout.addWidget(self.download_button)
            QMessageBox.information(self, "Éxito", "Procesamiento en lote completado.")
        else:
            QMessageBox.warning(self, "Advertencia", "No se guardara los datos procesados.")

    #Display results image 
    def show_error(self, message):
        """ Mostrar mensajes de error """
        QMessageBox.critical(self, "Error", message)
    
    def display_result(self, result_image):
        """Muestra la imagen procesada en el QLabel."""
        if result_image is None:
            self.image_widget.setText("No se pudo procesar la imagen.")
            self.image_widget.setStyleSheet("QLabel { color : red; font: 15px; }")
            return

        try:
            # Convertir la imagen a RGB si es necesario
            if len(result_image.shape) == 2:  # Si es escala de grises
                result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)
            elif result_image.shape[2] == 4:  # Si tiene canal alfa (RGBA)
                result_image = cv2.cvtColor(result_image, cv2.COLOR_RGBA2RGB)

            # Redimensionar la imagen para que se ajuste al QLabel
            height, width, channel = result_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(result_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Escalar la imagen para que se ajuste al tamaño del QLabel
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                self.image_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_widget.setPixmap(scaled_pixmap)
        except Exception as e:
            self.image_widget.setText(f"Error al mostrar la imagen: {str(e)}")
            self.image_widget.setStyleSheet("QLabel { color : red; font: 15px; }")

    def download_csv(self, df):
        """Guarda el DataFrame en un archivo CSV."""
        if df.empty:
            QMessageBox.warning(self, "Error", "No hay datos para guardar.")
            return

        # Abrir diálogo para seleccionar ubicación
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar CSV",
            "",
            "Archivos CSV (*.csv);;Todos los archivos (*)",
            options=options
        )

        if file_path:
            try:
                # Asegurar que la extensión sea .csv
                if not file_path.endswith('.csv'):
                    file_path += '.csv'
                
                # Guardar el DataFrame
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Éxito", f"Archivo guardado en:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo guardar el archivo:\n{str(e)}")
    def update_elapsed_time(self):
        """Actualiza el tiempo transcurrido cada segundo en la GUI."""
        if self.start_time is None:
            return  # Evita errores si el tiempo de inicio aún no está definido
        
        elapsed = time.time() - self.start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        # Mantener la información del tiempo transcurrido sin afectar el progreso
        current_text = self.time_label.text()
        if "Restante" in current_text:
            remaining_part = current_text.split("\n")[-1]  # Mantiene la parte del tiempo restante
            self.time_label.setText(f"Tiempo transcurrido: {elapsed_str}\n{remaining_part}")
        else:
            self.time_label.setText(f"Tiempo transcurrido: {elapsed_str}")

        QApplication.processEvents()  # Forzar actualización de la interfaz


class AnalyzePage(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.file_paths = {} 
        self.df = None
        
        
    def init_ui(self):
        self.setStyleSheet("background-color: #e8f8f5;")

        # 🔹 Gráfico (simulación con QLabel)
        graph_frame = QFrame(self)
        graph_frame.setStyleSheet("background-color: #fff; border: 1px solid #ccc;")
        graph_frame.setMinimumHeight(200)  # Aumentamos la altura para mayor espacio del gráfico

        graph_layout = QVBoxLayout(graph_frame)

        # Creamos la figura de Matplotlib y el canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        graph_layout.addWidget(self.canvas)
  
        self.toolbar = NavigationToolbar(self.canvas, self)
        graph_layout.addWidget(self.toolbar)
       

       # 🔹 Controles Inferiores
        control_layout = QHBoxLayout()  # Cambiamos a QHBoxLayout para dividir en dos columnas
       

        # ========== 📂 Panel Izquierdo ==========
        left_panel = QVBoxLayout()  # Panel Izquierdo


         # Botones en una fila

        # Caja para mostrar archivos cargados
        self.file_list = QListWidget()
        self.file_list.setFixedHeight(80)  # Ajuste de altura
        self.file_list.setFixedWidth(600)
        left_panel.addWidget(self.file_list)


        file_controls = QHBoxLayout()
        
        self.delete_file_button = QPushButton("Eliminar")
        self.delete_file_button.setStyleSheet("""
                QPushButton {
                    background-color: #D9534F;
                    color: white;
                    font-weight: bold;
                    font-size: 14px;
                    padding: 10px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #C9302C;
                }
            """)
        self.delete_file_button.clicked.connect(self.delete_file)
        file_controls.addWidget(self.delete_file_button)

        self.add_file_button = QPushButton("Agregar")
        self.add_file_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    font-size: 14px;
                    padding: 10px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #358338;
                }
            """)
        self.add_file_button.clicked.connect(self.add_file)
        file_controls.addWidget(self.add_file_button)
        left_panel.addLayout(file_controls)

    
        # Create a horizontal layout for the first scene coordinate value
        scene_coord_layout = QHBoxLayout()
        scene_coord_layout.addWidget(QLabel("Valor de la coordenada"))
        self.scene_coordinate_value = QDoubleSpinBox()
        self.scene_coordinate_value.setFixedWidth(100)
        self.scene_coordinate_value.setRange(-90, 90)
        self.scene_coordinate_value.setValue(-30)
        self.scene_coordinate_value.setDecimals(5)
        self.scene_coordinate_value.setStyleSheet("""
        QDoubleSpinBox {
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 5px;
            font-size: 14px;
            background-color: white;
        }
         """)
        scene_coord_layout.addWidget(self.scene_coordinate_value)
        left_panel.addLayout(scene_coord_layout)

        # Create a horizontal layout for the limit scene coordinate value
        limit_coord_layout = QHBoxLayout()
        limit_coord_layout.addWidget(QLabel("Rango por imagen de coordenada"))
        self.limit_scene_coordinate = QDoubleSpinBox()
        self.limit_scene_coordinate.setFixedWidth(100)
        self.limit_scene_coordinate.setRange(-90, 90)
        self.limit_scene_coordinate.setDecimals(4)
        self.limit_scene_coordinate.setValue(0.5000)
        self.limit_scene_coordinate.setStyleSheet("""
        QDoubleSpinBox {
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 5px;
            font-size: 14px;
            background-color: white;
        }
         """)
        
        limit_coord_layout.addWidget(self.limit_scene_coordinate)
        left_panel.addLayout(limit_coord_layout)
       

        converter_layout = QHBoxLayout()  # Layout para el panel de conversión



        self.Merges_button = QPushButton("Combinar")
        self.Merges_button.setStyleSheet("""
            QPushButton {
                background-color: #478EDB; /* Verde */
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3267A0;
            }
            """)
        self.Merges_button.clicked.connect(self.merge_files)
        converter_layout.addWidget(self.Merges_button)
        

        self.converter_file = QPushButton("Tranformar Archivo")
        self.converter_file.setStyleSheet("""
            QPushButton {
                background-color: #3CAAC5; /* Verde */
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2B7B8F;
            }
            """)
        self.converter_file.clicked.connect(self.converter)
        converter_layout.addWidget(self.converter_file)

        left_panel.addLayout(converter_layout)

        self.export_button = QPushButton("EXPORTAR CSV")
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #F0AD4E; /* Verde */
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #AB7D3A;
            }
            """)
        self.export_button.clicked.connect(self.export_csv)
        left_panel.addWidget(self.export_button)

        
        control_layout.addLayout(left_panel)  # Agregamos el panel izquierdo

        
        # ========== 📊 Panel Derecho ==========
        right_panel = QVBoxLayout()  # Panel Derecho



         # ComboBox para seleccionar el eje X
        self.combo_x = QComboBox()
        self.combo_x.setStyleSheet("""
            QComboBox {
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 5px;
            font-size: 14px;
            }
            QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left-width: 1px;
            border-left-color: darkgray;
            border-left-style: solid;
            border-top-right-radius: 3px;
            border-bottom-right-radius: 3px;
            }
            QComboBox::down-arrow {
            image: url(down_arrow.png);  /* Reemplaza con la ruta de tu imagen de flecha */
            width: 10px;
            height: 10px;
            }
        """)
        right_panel.addWidget(QLabel("Select X-axis:"))
        right_panel.addWidget(self.combo_x)

        # ComboBox para seleccionar el eje Y
        self.combo_y = QComboBox()
        self.combo_y.setStyleSheet("""
                        QComboBox {
                        background-color: #f0f0f0;
                        border: 1px solid #ccc;
                        border-radius: 4px;
                        padding: 5px;
                        font-size: 14px;
                        }
                        QComboBox::drop-down {
                        subcontrol-origin: padding;
                        subcontrol-position: top right;
                        width: 20px;
                        border-left-width: 1px;
                        border-left-color: darkgray;
                        border-left-style: solid;
                        border-top-right-radius: 3px;
                        border-bottom-right-radius: 3px;
                        }
                        QComboBox::down-arrow {
                        image: url(down_arrow.png);  /* Reemplaza con la ruta de tu imagen de flecha */
                        width: 10px;
                        height: 10px;
                        }
                    """)

        right_panel.addWidget(QLabel("Seleccionar Y-axis:"))
        right_panel.addWidget(self.combo_y)

        # ComboBox para seleccionar el color
        self.combo_color = QComboBox()

        self.combo_color.setStyleSheet("""
            QComboBox {
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 5px;
            font-size: 14px;
            }
            QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left-width: 1px;
            border-left-color: darkgray;
            border-left-style: solid;
            border-top-right-radius: 3px;
            border-bottom-right-radius: 3px;
            }
            QComboBox::down-arrow {
            image: url(down_arrow.png);  /* Reemplaza con la ruta de tu imagen de flecha */
            width: 10px;
            height: 10px;
            }
        """)
        right_panel.addWidget(QLabel("Seleccionar color:"))
        right_panel.addWidget(self.combo_color)

        # ComboBox para seleccionar el color
        self.filter = QHBoxLayout()
        self.filter_column = QComboBox()
        

        self.filter_column.setStyleSheet("""
            QComboBox {
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 5px;
            font-size: 14px;
            }
            QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left-width: 1px;
            border-left-color: darkgray;
            border-left-style: solid;
            border-top-right-radius: 3px;
            border-bottom-right-radius: 3px;
            }
            QComboBox::down-arrow {
            image: url(down_arrow.png);  /* Reemplaza con la ruta de tu imagen de flecha */
            width: 10px;
            height: 10px;
            }
        """)
        self.filter.addWidget(QLabel("Filtrar por:"))
        self.filter.addWidget(self.filter_column)


        # Create a QListWidget for multi-selection with checkboxes.
        self.filter_value = QListWidget()
        self.filter_value.setSelectionMode(QListWidget.MultiSelection)
        self.filter_value.setStyleSheet("""
            QListWidget {
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 5px;
            font-size: 14px;
            }
        """)
        # Example: populate with sample items (in practice, fill dynamically)
        for item_text in ["Opcion 1", "Opcion 2", "Opcion 3"]:
            item = QListWidgetItem(item_text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.filter_value.addItem(item)
        
        self.filter.addWidget(QLabel("Valores a filtrar:"))
        self.filter.addWidget(self.filter_value)



        right_panel.addLayout(self.filter)  # Cambia addWidget por addLayout


        # ComboBox para seleccionar el tipo de gráfico
        self.combo_chart_type = QComboBox()
        self.combo_chart_type.setStyleSheet("""
            QComboBox {
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 5px;
            font-size: 14px;
            }
            QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left-width: 1px;
            border-left-color: darkgray;
            border-left-style: solid;
            border-top-right-radius: 3px;
            border-bottom-right-radius: 3px;
            }
            QComboBox::down-arrow {
            image: url(down_arrow.png);  /* Reemplaza con la ruta de tu imagen de flecha */
            width: 10px;
            height: 10px;
            }
        """)
        self.combo_chart_type.addItems(["Scatter", "Bar", "Line", "Contour"])
        right_panel.addWidget(QLabel("Seleccionar tipo de grafica:"))
        right_panel.addWidget(self.combo_chart_type)

        # Botones de análisis y exportación
        #right_panel.addWidget(self.create_button("Choose the data", self.choose_data))


        self.analyze_button = QPushButton("Analizar")
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; /* Verde */
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            """)
        self.analyze_button.clicked.connect(self.generate_graph)
        right_panel.addWidget(self.analyze_button)

        self.download = QPushButton("Exportar Grafica")
        self.download.setStyleSheet("""
            QPushButton {
                background-color: #6F42C1; /* Verde */
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #51308D;
            }
            """)
        self.download.clicked.connect(self.export_graph)
        right_panel.addWidget(self.download)

        control_layout.addLayout(right_panel)  # Agregamos el panel derecho

         # En tu método init_ui o después de crear self.file_list
        self.file_list.currentItemChanged.connect(self.update_comboboxes)
        
        # 🔹 Layout Principal
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(graph_frame)
        main_layout.addLayout(control_layout)
        self.setLayout(main_layout)




    def update_comboboxes(self):
        # Actualiza los ComboBox con las columnas del DataFrame
        selected_item = self.file_list.currentItem()
        if selected_item:
            file_name = selected_item.text()
            file_path = self.file_paths.get(file_name)
            if file_path:
                try:
                    df = pd.read_csv(file_path)
                    self.df = df  # Actualizar self.df con el DataFrame actual
                    columns = df.columns.tolist()

                    self.combo_x.clear()
                    # Agregamos "None" antes de las columnas
                    self.combo_x.addItem("None")
                    self.combo_x.addItems(columns)
                    # Por defecto, dejamos seleccionado "None"
                    self.combo_x.setCurrentIndex(0)

                    self.combo_y.clear()
                    self.combo_y.addItem("None")
                    self.combo_y.addItems(columns)
                    self.combo_y.setCurrentIndex(0)

                    self.filter_column.clear()
                    self.filter_column.addItem("None")
                    self.filter_column.addItems(columns)
                    self.filter_column.setCurrentIndex(0)


                    self.filter_column.clear()
                    self.filter_column.addItem("None")
                    self.filter_column.addItems(columns)
                    self.filter_column.setCurrentIndex(0)


                    self.combo_color.clear()
                    self.combo_color.addItem("None")
                    self.combo_color.addItems(columns)
                    self.combo_color.setCurrentIndex(0)

                  
                    self.filter_column.currentTextChanged.connect(self.update_filter_values)
                    self.update_filter_values()
                                
        
                   
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"No se pudo leer el archivo:\n{str(e)}")
            else:
                QMessageBox.warning(self, "Error", "No se pudo encontrar la ruta del archivo seleccionado.")
        else:
            QMessageBox.warning(self, "Advertencia", "No hay ningún archivo seleccionado en la lista.")

    def update_filter_values(self):
        """Actualiza el combobox de valores únicos basándose en la columna seleccionada"""
        self.filter_value.clear()
        self.filter_value.addItem("None")

        selected_column = self.filter_column.currentText()

        if selected_column != "None" and selected_column in self.df.columns:
            unique_values = self.df[selected_column].dropna().unique()
            self.filter_value.addItems(map(str, unique_values))

            
    def generate_graph(self):
        if self.df is not None:
            try:
                print("Generando gráfico...")
                # Obtener valores de los combos y eliminar espacios en blanco
                x = self.combo_x.currentText().strip()
                y = self.combo_y.currentText().strip()
                color = self.combo_color.currentText().strip()
                filter_column_name = self.filter_column.currentText().strip()  # Nombre de la columna a filtrar
                filter_values =  [item.text().strip() for item in self.filter_value.selectedItems()]
         # Valor a filtrar
                chart_type = self.combo_chart_type.currentText().strip()
                
                print(f"Seleccionado - X: {x}, Y: {y}, Color: {color}, Filter Column: {filter_column_name}, Filter Value: {filter_values}, Tipo de Gráfico: {chart_type}")

                # Validar que las columnas x e y existan
                for col in [x, y]:
                    if col == "None" or col not in self.df.columns:
                        QMessageBox.warning(self, "Error", f"La columna '{col}' no existe en el DataFrame.")
                        return

                # Definir el DataFrame filtrado
                if filter_column_name != "None" and filter_values and "None" not in filter_values:
                    filtered_df = self.df[self.df[filter_column_name].astype(str).isin(filter_values)]
                else:
                    filtered_df = self.df


                # Verificar si hay valores nulos en x o y
                if self.df[[x, y]].isnull().values.any():
                    QMessageBox.warning(self, "Advertencia", "Las columnas seleccionadas contienen valores nulos.")
                    return

                # Limpiar la figura actual y crear un nuevo subplot
                self.figure.clear()
                ax = self.figure.add_subplot(111)

                # Construir el gráfico según el tipo seleccionado
                if chart_type == "Scatter":
                    if color != "None" and color in self.df.columns:
                        sns.scatterplot(data=filtered_df, x=x, y=y, hue=color, ax=ax)
                    else:
                        sns.scatterplot(data=filtered_df, x=x, y=y, ax=ax)
                elif chart_type == "Bar":
                    if color != "None" and color in self.df.columns:
                        sns.barplot(data=filtered_df, x=x, y=y, hue=color, ax=ax)
                    else:
                        sns.barplot(data=filtered_df, x=x, y=y, ax=ax)
                elif chart_type == "Line":
                    if color != "None" and color in self.df.columns:
                        sns.lineplot(data=filtered_df, x=x, y=y, hue=color, ax=ax)
                    else:
                        sns.lineplot(data=filtered_df, x=x, y=y, ax=ax)
                elif chart_type == "Pie":
                    # El gráfico de pastel requiere que 'color' (la columna para agrupar) sea válido
                    if color == "None" or color not in self.df.columns:
                        QMessageBox.warning(self, "Error", "El gráfico de pastel requiere una columna válida para 'Color'.")
                        return
                    data = filtered_df.groupby(color)[y].sum() if filter_column_name != "None" else self.df.groupby(color)[y].sum()
                    data.plot.pie(autopct='%1.1f%%', ax=ax)
                    ax.set_ylabel('')
                    ax.set_xlabel('')
                elif chart_type == "Contour":
                    if x in self.df.columns and y in self.df.columns:
                        x_vals = self.df[x].values
                        y_vals = self.df[y].values
                        sns.kdeplot(x=x_vals, y=y_vals, fill=True, ax=ax, cmap="viridis")
                    else:
                        QMessageBox.warning(self, "Error", "Las columnas seleccionadas no existen en el DataFrame.")
                        return
                else:
                    QMessageBox.warning(self, "Error", f"Tipo de gráfico no soportado: {chart_type}")
                    return

                # Configurar etiquetas y leyenda
                ax.set_title('Generated Graph')
                if chart_type != "Pie":
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    if color != "None" and color in self.df.columns:
                        ax.legend(title=color)
                    else:
                        ax.legend()
                else:
                    ax.legend()

                self.canvas.draw()
                print("Gráfico generado correctamente.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Ocurrió un error al generar el gráfico:\n{str(e)}")
                print(f"Error al generar el gráfico: {str(e)}")
        else:
            QMessageBox.warning(self, "Advertencia", "No hay datos disponibles para graficar.")

    def export_graph(self):
        if self.df is not None:
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Graph",
                "",
                "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf)"
            )
            if save_path:
                self.canvas.draw()  # Forzar actualización del canvas
                self.figure.savefig(save_path)
                QMessageBox.information(self, "Éxito", f"Gráfico guardado en:\n{save_path}")
        else:
            QMessageBox.warning(self, "Advertencia", "No hay datos disponibles para exportar el gráfico.")



    def create_button(self, text, callback):
        btn = QPushButton(text)
        btn.setStyleSheet("background-color: #005f80; color: white; padding: 5px; font-weight: bold;")
        btn.clicked.connect(callback)
        return btn

    def converter(self):
        print("Convertir archivo")
        reply = QMessageBox.question(
            self, 'Confirmar conversión', 'Prodecera a convertir el archivo seleccionado. ¿Está seguro?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            print("Conversión confirmada")
           #Convert only if the following headers in the CSV file are present
            selected_item = self.file_list.currentItem()

            if not selected_item:
                QMessageBox.warning(self, "Advertencia", "No hay archivo seleccionado.")
                return
            
            

            file_name = selected_item.text()  # Nombre del archivo
            file_path = self.file_paths.get(file_name)  # Obtener la ruta completa
            
            if not file_path:
                QMessageBox.warning(self, "Error", f"No se encontró la ruta completa para: {file_name}")
                return

            # Verificar si el archivo realmente existe antes de leerlo
            if not os.path.exists(file_path):
                QMessageBox.warning(self, "Error", f"El archivo no existe: {file_path}")
                return

                # Verificar si tiene las columnas necesarias
            try:
                df = pd.read_csv(file_path, nrows=0)  # Leer solo los encabezados
                required_headers = {'Imagen_idobject', 'Porcentaje_no_pintado', 'Diametro_mm'}
                
                if required_headers.issubset(df.columns):
                    self.tranform = Tranform_files(file_path, 2,self.scene_coordinate_value.value(), self.limit_scene_coordinate.value())
                    self.tranform.path_converted.connect(self.show_converted)  # Conectar la señal
                    self.tranform.start()
                    QApplication.processEvents()  # Actualizar la interfaz
                    print("✅ Archivo convertido correctamente")

                else:
                    QMessageBox.warning(self, "Advertencia", "El archivo debe contener al menos 'Imagen_idobject', 'Diametro_mm' y 'Porcentaje_no_pintado'.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"No se pudo leer el archivo:\n{str(e)}")
                


        else:
            print("Conversión cancelada")

        # Placeholder for file conversion functionality
    def show_converted(self, converted_path):
        file_name = os.path.basename(converted_path)
        self.file_list.addItem(file_name)  # Agregar el archivo a la lista
        self.file_paths[file_name] = converted_path  # Guardar la ruta real


    def delete_file(self):
        print("Eliminar archivo")
        selected_item = self.file_list.currentItem()
        if selected_item:
            self.file_list.takeItem(self.file_list.row(selected_item))

    def add_file(self):
        print("Agregar archivo")

        # Add a file, selecting an existing file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select a file", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            file_name = os.path.basename(file_path)
            if file_name.endswith(".csv"):
                # Save the file in the same temporary directory
                if not hasattr(self, 'temp_dir'):
                    self.temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(self.temp_dir, file_name)
                shutil.copy(file_path, temp_file_path)
                
                self.file_list.addItem(file_name)
                self.file_paths[file_name] = temp_file_path
            else:
                print("Only CSV files are allowed")
        else:
            print("No file selected")
       


    def merge_files(self):
        # Se asume que se usa un diálogo para seleccionar múltiples archivos
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar archivos a combinar",
            "",  # O un directorio predeterminado
            "Archivos compatibles (*.txt *.csv);;Todos los archivos (*)"
        )
        
        if files:
            self.start_merge_process(files)
            print("Starting....")
        else:
            QMessageBox.warning(self, "Advertencia", "No se seleccionaron archivos para fusionar.")

    def start_merge_process(self, files):
        # Crear y configurar el hilo para merge, con type_process = 1
        self.merge_thread = Tranform_files(files, 1, 1, 1)  # start_number e increment según tus requerimientos
        self.merge_thread.error_signal.connect(self.show_error)
        self.merge_thread.success_signal.connect(self.show_success)
        self.merge_thread.start()


    def show_error(self, title, message):
        QMessageBox.critical(self, title, message, QMessageBox.Ok)

    def show_success(self, output_path, total_files, total_records):
        msg = f"""
        Archivos combinados exitosamente:
        - Total archivos: {total_files}
        - Total registros: {total_records}
        - Guardado en: {output_path}
        """
        QMessageBox.information(self, "Merge completado", msg, QMessageBox.Ok)



    def choose_data(self):
        print("Elegir datos")

    def export_csv(self):
        selected_item =self.file_list.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "Advertencia", "No hay archivo seleccionado.")
            return
        file_name = selected_item.text()
        file_path = self.file_paths.get(file_name)
        if not file_path:
            QMessageBox.warning(self, "Error", f"No se encontró la ruta completa para: {file_name}")
            return
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Error", f"El archivo no existe: {file_path}")
            return
        try:
            df = pd.read_csv(file_path)
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Guardar CSV", "", "CSV Files (*.csv);;All Files (*)"
            )
            if save_path:
                df.to_csv(save_path, index=False)
                QMessageBox.information(self, "Éxito", f"Archivo guardado en:\n{save_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"No se pudo leer el archivo:\n{str(e)}")


class HomePage(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        # Layout principal vertical
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Manejar ruta del icono correctamente
        if getattr(sys, 'frozen', False):
            # Si la aplicación está empaquetada con PyInstaller
            base_path = sys._MEIPASS
        else:
            # Si se ejecuta como un script normal
            base_path = os.path.abspath(".")

        icon_path = os.path.join(base_path, "icon_max.ico")
        
        # Logo o imagen (opcional)
        logo = QLabel()
        pixmap = QPixmap(icon_path)  # Reemplaza con la ruta de tu logo o imagen corporativa
        if not pixmap.isNull():
            logo.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logo.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(logo)
        
        # Etiqueta de bienvenida
        welcome_label = QLabel("Dector de sistemas modulares")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_font = QFont("Arial", 24, QFont.Bold)
        welcome_label.setFont(welcome_font)
        welcome_label.setStyleSheet("color: #2c3e50;")
        main_layout.addWidget(welcome_label)
        
        # Etiqueta descriptiva o de estado (por ejemplo, "en desarrollo")
        description_label = QLabel("Aplicación desarrollado para usar con Materiales de poliuretano")
        description_label.setAlignment(Qt.AlignCenter)
        description_font = QFont("Arial", 14)
        description_label.setFont(description_font)
        description_label.setStyleSheet("color: #34495e;")
        main_layout.addWidget(description_label)

        # Obtener rutas absolutas para los documentos
        pdf_path = os.path.abspath("Resources\Manual_Aplicacion_1.0.0.pdf")
        zip_path = os.path.abspath("Resources\Muestras.zip")


    

        # Obtener rutas absolutas para los documentos
        pdf_path = get_resource_path("Resources/Manual_Aplicacion_1.0.0.pdf")
        zip_path = get_resource_path("Resources/Muestras.zip")

        # Etiqueta para descargar el documento PDF usando el protocolo file:///
        pdf_label = QLabel(f'<a href="file:///{pdf_path}">Descargar Manual</a>')
        pdf_label.setAlignment(Qt.AlignCenter)
        pdf_label.setFont(description_font)
        pdf_label.setStyleSheet("color: #2980b9;")
        pdf_label.setOpenExternalLinks(True)
        main_layout.addWidget(pdf_label)

        # Etiqueta para descargar el archivo ZIP usando el protocolo file:///
        zip_label = QLabel(f'<a href="file:///{zip_path}">Descargar Ejemplos en ZIP</a>')
        zip_label.setAlignment(Qt.AlignCenter)
        zip_label.setFont(description_font)
        zip_label.setStyleSheet("color: #2980b9;")
        zip_label.setOpenExternalLinks(True)
        main_layout.addWidget(zip_label)




        
        # Agregar un espacio final para centrar verticalmente
        main_layout.addStretch()

      
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()  
        name_label = QLabel("Desarrollado por : Carlos Garcia Cano")  
        name_label.setFont(QFont("Arial", 5))
        name_label.setStyleSheet("color: #95a5a6;")
        bottom_layout.addWidget(name_label)
        main_layout.addLayout(bottom_layout)
        # Aplicar el layout y los estilos generales
        self.setLayout(main_layout)
        self.setStyleSheet("""
            QWidget {
                background-color: #ecf0f1;
            }
        """)


# ------------------ VENTANA PRINCIPAL -------------------
class MainApp(QMainWindow):
    def __init__(self):

                # Manejar ruta del icono correctamente
        if getattr(sys, 'frozen', False):
            # Si la aplicación está empaquetada con PyInstaller
            base_path = sys._MEIPASS
        else:
            # Si se ejecuta como un script normal
            base_path = os.path.abspath(".")

        icon_path = os.path.join(base_path, "icon_max.ico")
        super().__init__()
        self.setWindowTitle("Sistema Modular")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon(icon_path))
        
        # Configurar el sistema de páginas
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Crear instancias de las páginas
        self.home_page = HomePage()
        self.clean_page = CleanPage()
        self.process_page = ProcessPage()
        self.analyze_page = AnalyzePage()
        
        # Agregar páginas al stacked widget
        self.stacked_widget.addWidget(self.home_page)    # índice 0
        self.stacked_widget.addWidget(self.clean_page)   # índice 1
        self.stacked_widget.addWidget(self.process_page) # índice 2
        self.stacked_widget.addWidget(self.analyze_page) # índice 3
        
        # Configurar menú
        self.create_menu()
        
        

    def create_menu(self):
        menubar = self.menuBar()
        
        # Menú de navegación
        nav_menu = menubar.addMenu("Navegar")
        
        actions = [
            ("Inicio", 0),
            ("Limpiar", 1),
            ("Procesar", 2),
            ("Analizar", 3)
        ]
        
        for text, index in actions:
            action = QAction(text, self)
            action.triggered.connect(lambda _, idx=index: self.stacked_widget.setCurrentIndex(idx))
            nav_menu.addAction(action)
        
        # Menú de ayuda
        help_menu = menubar.addMenu("Ayuda")
        about_action = QAction("Acerca de", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)


    def show_about(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Acerca del Sistema")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(
            "Sistema Modular de Procesamiento<br>"
            "Versión 1.0.0<br><br>"
            "Módulos disponibles:<br>"
            "- Limpieza de imágenes<br>"
            "- Deteccion en imágenes<br>"
            "- Analisis de datos<br><br>"
            "Para más información, accede al manual proporcionado.<br><br>"
            "Desarrollado por Carlos Jesús García Cano<br>"
            'GitHub: <a href="https://github.com/CarlosGarcia-YK">https://github.com/CarlosGarcia-YK</a>'
        )
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

def get_resource_path(relative_path):
                    """ Obtiene la ruta real de los archivos cuando se ejecuta como .exe """
                    if getattr(sys, 'frozen', False):  # Si está empaquetado con PyInstaller
                        base_path = sys._MEIPASS
                    else:
                        base_path = os.path.abspath(".")
                    
                    return os.path.join(base_path, relative_path)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())