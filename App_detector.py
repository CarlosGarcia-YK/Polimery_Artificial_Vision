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
import random
import pandas as pd
import tempfile
from tqdm import tqdm
from PIL import Image
import time 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog, QSlider, QHBoxLayout,
    QVBoxLayout, QSplitter, QListWidget, QFormLayout, QSpinBox, QMessageBox, QProgressBar,QAction, qApp, QStackedWidget, QFileDialog, QMessageBox, QPushButton, QFrame, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer  
from PyQt5.QtGui import QPixmap, QImage, QFont
import shutil

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

    def emit_progress(self, start_time, current_stage, total_stages):
        # En el método donde inicias el procesamiento
      
        elapsed = time.time() - start_time
        avg_time_per_stage = elapsed / current_stage
        remaining_time = avg_time_per_stage * (total_stages - current_stage)
        progress = int((current_stage / total_stages) * 100)
        self.progress_updated.emit(progress, elapsed, remaining_time)

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

class CleaningProcessor(QThread): 
    progress_updated = pyqtSignal(int)  # Progreso del procesamiento
    image_processed = pyqtSignal(str)
    path_processed  = pyqtSignal(str)
    error_occurred = pyqtSignal(str)  # Señal de error
    processing_finished = pyqtSignal(bool)  # Señal de finalización del procesamiento
    
    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
        self.is_running = True
        self.avg_time_per_image = 0
        self.temp_dir = tempfile.mkdtemp()
       

         # Inicialmente, el procesamiento no ha terminado
      

    def run(self):
        try:
            self.start_time = time.time()
            total_images = len(self.image_paths)

            rect1_points = [(589, 860), (1002, 858), (590, 934), (1002, 933)]
            rect2_points = [(480, 114), (481, 2), (1, 2), (1, 114)]
            rect3_points = [(1178, 2), (1177, 251), (1470, 252), (1470, 2)]
            rect4_points = [(190, 724), (1, 723), (4, 934), (191, 931)]
            rect5_points = [(1325, 861), (1468, 864), (1468, 931), (1327, 933)]

            for i, img_path in enumerate(self.image_paths):
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                # Si la imagen está en escala de grises, convertirla a RGB
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                # Dibujar los rectángulos
                image = self.draw_rectangle_from_points(image, rect1_points)
                image = self.draw_rectangle_from_points(image, rect2_points)
                image = self.draw_rectangle_from_points(image, rect3_points)
                image = self.draw_rectangle_from_points(image, rect4_points)
                image = self.draw_rectangle_from_points(image, rect5_points)

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


    def __init__(self, path_origin, type_process,start_number, increment):
            super().__init__()
            self.path_origin = path_origin
            self.start_number = start_number
            self.increment = increment
            self.type_process = type_process
            self.temp_dir = tempfile.mkdtemp()
            
    def run(self):
        try:
            if self.type_process == 1:
                self.merge_files(self.path_origin)
             
            if self.type_process == 2:
               self.convert_files(self.path_origin,self.start_number, self.increment)
                
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")
    """Combina archivos manteniendo cada uno en tablas separadas"""
    """
    def merge_files(self):
       
    
        try:
            # Configurar opciones del diálogo
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.ExistingFiles)
            file_dialog.setNameFilter("Archivos compatibles (*.txt *.csv)")
            
            if file_dialog.exec_():
                files = file_dialog.selectedFiles()
                
                # Validar que se seleccionaron archivos
                if not files:
                    QMessageBox.warning(self, "Advertencia", "No se seleccionaron archivos", QMessageBox.Ok)
                    return
                    
                # Validar tipos de archivo
                extensions = {os.path.splitext(f)[1].lower() for f in files}
                if len(extensions) != 1 or extensions.pop() not in ['.txt', '.csv']:
                    QMessageBox.critical(self, "Error", "Todos los archivos deben ser del mismo tipo (.txt o .csv)", QMessageBox.Ok)
                    return

                # Determinar el tipo de archivo
                file_type = os.path.splitext(files[0])[1].lower()
                
                # Procesar cada archivo
                for i, file_path in enumerate(files, start=1):
                    try:
                        # Leer archivo
                        if file_type == '.csv':
                            df = pd.read_csv(file_path)
                        else:
                            df = pd.read_csv(file_path, delimiter='\t')
                            
                        # Generar nombre único para la tabla
                        base_name = os.path.basename(file_path)
                        table_name = f"tabla_{i}_{os.path.splitext(base_name)[0]}"
                        df['Nombre_archivo'] = os.path.basename(file_path)
                        
                        # Guardar en nueva tabla (implementación específica)
                        self.save_to_table(df, table_name)
                        
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Error procesando {os.path.basename(file_path)}:\n{str(e)}", QMessageBox.Ok)
                        continue
                        
                # Notificar finalización
                QMessageBox.information(self, "Éxito", f"{len(files)} archivos convertidos a tablas", QMessageBox.Ok)


                
        except Exception as e:
            QMessageBox.critical(self, "Error crítico", f"Error inesperado: {str(e)}", QMessageBox.Ok)
        finally:
            # Limpiar recursos si es necesario
            pass

    def save_to_table(self, dataframe, table_name):
        # Aquí iría tu lógica específica de guardado
        print(f"Guardando {table_name} con {len(dataframe)} registros")
        
        # Ejemplo: Guardar como nuevo CSV
        output_path = f"{table_name}.csv"
        dataframe.to_csv(output_path, index=False)
        
        # O guardar en base de datos, etc.
    """


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
            count_table.to_csv(output_filename, index=False)
            print(f"✅ Archivo convertido guardado en: {output_filename}")

            self.path_converted.emit(output_filename)
            return output_filename

        except Exception as e:
            print(f"❌ Error crítico: {str(e)}")
            raise


class CleanPage(QWidget):
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
        self.setStyleSheet("background-color: #fff9e6;")

        # Panel izquierdo con controles
        self.control_widget = QWidget()
        self.control_widget.setLayout(self.control_layout)

        self.image_widget = QLabel()
        self.image_widget.setFixedSize(800, 800)
        self.image_widget.setAlignment(Qt.AlignCenter)
        self.image_widget.setText("Limpieza de imagenes.")
        self.image_widget.setStyleSheet("QLabel { color : gray; font: 18px; }")

        # Botón para seleccionar directorio de imágenes
        self.select_dir_button = QPushButton("Seleccionar Directorio de Imágenes")
        self.select_dir_button.clicked.connect(self.select_directory)
        self.control_layout.addWidget(self.select_dir_button)


        # Lista de imágenes
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.image_selected)
        self.control_layout.addWidget(self.image_list)

        # Botón para procesar todas las imágenes
        self.process_all_button = QPushButton("Procesar Todas las Imágenes")
        self.process_all_button.clicked.connect(self.convert_images)
        self.control_layout.addWidget(self.process_all_button)

        self.save_button = QPushButton("Guardar Imagen", self)
        
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

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.control_widget)
        self.splitter.addWidget(self.image_widget)

       
        # 🔹 Crear el layout principal
        self.main_layout = QHBoxLayout(self)
        self.main_layout.addWidget(self.splitter)
        self.setLayout(self.main_layout)  # Asignar el layout principal a la ventana
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
        
       
            
        # Configurar y lanzar el hilo
        self.processor = CleaningProcessor(self.image_paths)
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(self.image_widget.size(), Qt.KeepAspectRatio)
            self.image_widget.setPixmap(scaled_pixmap)
        else:
            self.image_widget.setText("Imagen no válida o no se pudo cargar.")
            self.image_widget.setStyleSheet("QLabel { color : red; font: 15px; }")
    

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
        self.setStyleSheet("background-color: #fff9e6;")

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
        self.select_dir_button.clicked.connect(self.select_directory)
        self.control_layout.addWidget(self.select_dir_button)

        # Lista de imágenes
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.image_selected)
        self.control_layout.addWidget(self.image_list)

        # Controles de parámetros
        self.parameters_layout = QFormLayout()

        #CLAHE clip limit
        self.clip_limit_spin = QSpinBox()
        self.clip_limit_spin.setRange(1, 10)
        self.clip_limit_spin.setValue(6)
        self.parameters_layout.addRow("CLAHE clip limit:", self.clip_limit_spin)

        #CLAHE grid size
        self.grid_size_spin = QSpinBox()
        self.grid_size_spin.setRange(1, 16)
        self.grid_size_spin.setValue(16)
        self.parameters_layout.addRow("CLAHE grid size:", self.grid_size_spin)

        #Suaviazdo de imagen
        self.blur_size_spin = QSpinBox()
        self.blur_size_spin.setRange(1, 15)
        self.blur_size_spin.setValue(5)
        self.blur_size_spin.setSingleStep(2)
        self.parameters_layout.addRow("Tamaño de suavizado:", self.blur_size_spin)

        
        # Tamaño del kernel
        self.kernel_size_spin = QSpinBox()
        self.kernel_size_spin.setRange(1, 15)
        self.kernel_size_spin.setValue(3)
        self.kernel_size_spin.setSingleStep(2)
        self.parameters_layout.addRow("Tamaño del filtro de detección:", self.kernel_size_spin)

        # Iteraciones morfológicas
        self.morph_iterations_spin = QSpinBox()
        self.morph_iterations_spin.setRange(1, 10)
        self.morph_iterations_spin.setValue(2)
        self.parameters_layout.addRow("Número de ajustes:", self.morph_iterations_spin)

        # Distancia mínima entre picos
        self.min_distance_peak_spin = QSpinBox()
        self.min_distance_peak_spin.setRange(1, 20)
        self.min_distance_peak_spin.setValue(8)
        self.parameters_layout.addRow("Separación mínima entre puntos detectados:", self.min_distance_peak_spin)

        # Agregar los controles de parámetros al layout
        self.control_layout.addLayout(self.parameters_layout)

        # Botón para procesar imagen
        self.process_image_button = QPushButton("Procesar Imagen")
        self.process_image_button.clicked.connect(self.process_image)
        self.control_layout.addWidget(self.process_image_button)

        # Botón para procesar todas las imágenes
        self.process_all_button = QPushButton("Procesar Todas las Imágenes")
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
            self.download_button = QPushButton("Descargar CSV", self)
            self.download_button.clicked.connect(lambda: self.download_csv(self.results_df))
            self.control_layout.addWidget(self.download_button)
            QMessageBox.information(self, "Éxito", "Procesamiento en lote completado.")
        else:
            QMessageBox.warning(self, "Advertencia", "No se detectaron burbujas válidas en las imágenes.")

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
        
    def init_ui(self):
        self.setStyleSheet("background-color: #e8f8f5;")

        # 🔹 Gráfico (simulación con QLabel)
        graph_frame = QFrame(self)
        graph_frame.setStyleSheet("background-color: #fff; border: 1px solid #ccc;")
        graph_frame.setMinimumHeight(200)

        graph_layout = QVBoxLayout(graph_frame)
        graph_label = QLabel("Gráfica de burbujas", graph_frame)
        graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        graph_label.setStyleSheet("font-size: 20px; color: #333; font-weight: bold;")
        graph_layout.addWidget(graph_label)

        

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
        file_controls.addWidget(self.create_button("Deleted", self.delete_file))
        file_controls.addWidget(self.create_button("Add", self.add_file))
        left_panel.addLayout(file_controls)

    
        # Create a horizontal layout for the first scene coordinate value
        scene_coord_layout = QHBoxLayout()
        scene_coord_layout.addWidget(QLabel("First Scene_coordinate value"))
        self.scene_coordinate_value = QDoubleSpinBox()
        self.scene_coordinate_value.setFixedWidth(100)
        self.scene_coordinate_value.setRange(-90, 90)
        self.scene_coordinate_value.setValue(-30)
        self.scene_coordinate_value.setDecimals(4)
        scene_coord_layout.addWidget(self.scene_coordinate_value)
        left_panel.addLayout(scene_coord_layout)

        # Create a horizontal layout for the limit scene coordinate value
        limit_coord_layout = QHBoxLayout()
        limit_coord_layout.addWidget(QLabel("Limit Scene_coordinate value"))
        self.limit_scene_coordinate = QDoubleSpinBox()
        self.limit_scene_coordinate.setFixedWidth(100)
        self.limit_scene_coordinate.setRange(-90, 90)
        self.limit_scene_coordinate.setDecimals(4)
        self.limit_scene_coordinate.setValue(0.5000)
        limit_coord_layout.addWidget(self.limit_scene_coordinate)
        left_panel.addLayout(limit_coord_layout)
       

        converter_layout = QHBoxLayout()  # Layout para el panel de conversión
        converter_layout.addWidget(self.create_button("Merges", self.merge_files))
        converter_layout.addWidget(self.create_button("Convert File", self.converter))
        left_panel.addLayout(converter_layout)

        left_panel.addWidget(self.create_button("EXPORT CSV", self.export_csv))

        

        control_layout.addLayout(left_panel)  # Agregamos el panel izquierdo


        # ========== 📊 Panel Derecho ==========
        right_panel = QVBoxLayout()  # Panel Derecho

        # Botones de variables en una fila
        var_layout = QHBoxLayout()
        var_layout.addWidget(self.create_button("V1", self.select_variables))
        var_layout.addWidget(self.create_button("V2", self.select_variables))
        var_layout.addWidget(self.create_button("V3", self.select_variables))
        right_panel.addLayout(var_layout)

        # Botones en columna
        right_panel.addWidget(self.create_button("Choose the data", self.choose_data))
        right_panel.addWidget(self.create_button("Analyze", self.generate_graph))  # Cambiado el nombre
        

        control_layout.addLayout(right_panel)  # Agregamos el panel derecho

        
        # 🔹 Layout Principal
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(graph_frame)
        main_layout.addLayout(control_layout)
        self.setLayout(main_layout)

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

        #Add a file, selecting an existing file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select a file", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            file_name = os.path.basename(file_path)
            if file_name.endswith(".csv"):
                self.file_list.addItem(file_name)
                self.file_paths[file_name] = file_path
            else :
                print("only CSV files are allowed")
            
        else:
            print("No file selected")
        
       


    def merge_files(self):
        print("Combinar archivos")

    def select_variables(self):
        print("Seleccionar variables")

    def choose_data(self):
        print("Elegir datos")

    def generate_graph(self):
        print("Generar gráfica")

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
        layout = QVBoxLayout()
        label = QLabel("Página Principal - Bienvenido")
        label.setFont(QFont('Arial', 18))
        label.setStyleSheet("color: #2c3e50;")
        
        layout.addStretch()
        layout.addWidget(label)
        layout.addStretch()
        
        self.setLayout(layout)
        self.setStyleSheet("background-color: #ecf0f1;")


# ------------------ VENTANA PRINCIPAL -------------------
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema Modular")
        self.setGeometry(100, 100, 1200, 800)
        
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
            ("Home", 0),
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
        QMessageBox.information(
            self,
            "Acerca del Sistema",
            "Sistema Modular de Procesamiento\n"
            "Versión 4.0\n\n"
            "Módulos disponibles:\n"
            "- Limpieza de imágenes\n"
            "- Procesamiento de datos\n"
            "- Análisis avanzado",
            QMessageBox.Ok
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())