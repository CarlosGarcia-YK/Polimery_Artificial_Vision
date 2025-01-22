import sys
import os
import cv2
import skimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from pandas import value_counts
import numpy as np
import math
import random
import pandas as pd
from tqdm import tqdm
import time 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog, QSlider, QHBoxLayout,
    QVBoxLayout, QSplitter, QListWidget, QFormLayout, QSpinBox, QMessageBox, QProgressBar,QAction
)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer  
from PyQt5.QtGui import QPixmap, QImage


#QThread para procesar las imágenes
class ImageProcessor(QThread):
    # Señales para comunicación con la GUI
    progress_updated = pyqtSignal(int, float, float)         # Progreso del procesamiento
    image_processed = pyqtSignal(np.ndarray, int)  # Imagen resultante y contador
    batch_finished = pyqtSignal(pd.DataFrame)      # Resultados finales del lote
    error_occurred = pyqtSignal(str)  
    
                 # Mensajes de error

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
                result, _, _, batch_data = self.process_single_image(img_path, self.params)
                valid_count = len(batch_data) if batch_data else 0
                if batch_data is not None:
                    if total_images > 1:# Si es solo una imagen, emitir el resultado
                         all_batch_data.extend(batch_data)  # Acumular datos válidos
                if result is not None:
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
            total_stages = 11  # Total de etapas definidas
            current_stage = 0
            
            # Etapa 1: Cargar imagen
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None, None, None, None
            current_stage += 1
            #self.emit_progress(start_time, current_stage, total_stages)

            # Etapa 2: Suavizar
            blur = cv2.GaussianBlur(image, (3, 3), 0)
            current_stage += 1
            #self.emit_progress(start_time, current_stage, total_stages)

            # Etapa 3: Umbral Otsu
            _, binary = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.bitwise_not(binary)
            current_stage += 1
            #self.emit_progress(start_time, current_stage, total_stages)

            # Etapa 4: Morfología
            kernel_size = params.get('kernel_size', 3)
            morph_iterations = params.get('morph_iterations', 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            binary_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
            current_stage += 1
            #self.emit_progress(start_time, current_stage, total_stages)

            # Etapa 5: Transformada distancia
            dist_transform = cv2.distanceTransform(binary_opened, cv2.DIST_L2, 5)
            current_stage += 1
            #self.emit_progress(start_time, current_stage, total_stages)

            # Etapa 6: Picos locales
            min_distance_peak = params.get('min_distance_peak', 8)
            coordinates = peak_local_max(dist_transform, min_distance=min_distance_peak, threshold_abs=0.5)
            local_max = np.zeros(dist_transform.shape, dtype=bool)
            local_max[tuple(coordinates.T)] = True
            current_stage += 1
            #self.emit_progress(start_time, current_stage, total_stages)

            # Etapa 7: Etiquetado
            markers, _ = ndimage.label(local_max)
            current_stage += 1
            #self.emit_progress(start_time, current_stage, total_stages)

            # Etapa 8: Watershed
            labels = watershed(-dist_transform, markers, mask=binary_opened)
            current_stage += 1
            #self.emit_progress(start_time, current_stage, total_stages)

            # Etapa 9: Cálculo porcentaje
            total_pixels = binary_opened.size
            painted_pixels = np.count_nonzero(binary_opened)
            unpainted_percentage = ((total_pixels - painted_pixels) / total_pixels) * 100
            unpainted_percentage_adjusted = max(0, unpainted_percentage - 15.35)
            current_stage += 1
            #self.emit_progress(start_time, current_stage, total_stages)
            

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
            #self.emit_progress(start_time, current_stage, total_stages)

            # Etapa 11: Procesamiento final
            colored_result, batch_data, valid_count = self.process_bubbles(
                labels, coordinates, params, unpainted_percentage_adjusted, image_path
            )
            #self.emit_progress(start_time, total_stages, total_stages)  # 100%
            
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

    def process_bubbles(self, labels, coordinates, params, unpainted_percentage, image_path):
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

            if 0.7 <= aspect_ratio < 2.0 and params['min_area_threshold'] <= area_px <= params['max_area_threshold']:
                # Cálculos de métricas
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                perimeter_px = cv2.arcLength(contours[0], True) if contours else 0.0
                diametro_px = math.sqrt((4 * area_px) / math.pi)
                class_id = self.classify_by_size(area_px, increase=150)  # Llamada corregida
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





# Clase principal de la aplicación
class MainWindow(QMainWindow):
    #Only the values of the Window
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detección de Burbujas")
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()
        self.image_paths = []
        self.current_image_path = None
        self.batch_data = []
        self.params = {}
        self.result_image = None
        self.processor = None
        self.start_time = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_elapsed_time)
    #Values and buttons of the menu
    def initUI(self):
     # Create a menu bar
        menu_bar = self.menuBar()

        # Add "File" menu
        file_menu = menu_bar.addMenu("File")

        # Add actions to "File" menu
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Add "Mode" menu
        mode_menu = menu_bar.addMenu("Mode")

        # Add actions to "Mode" menu
        clean_action = QAction("Clean Image", self)
        clean_action.triggered.connect(self.clean_image)
        mode_menu.addAction(clean_action)

        process_action = QAction("Process Detection", self)
        process_action.triggered.connect(self.process_detection)
        mode_menu.addAction(process_action)

        graph_action = QAction("Graphing", self)
        graph_action.triggered.connect(self.graphing)
        mode_menu.addAction(graph_action)

       
        """1.0.0 Version base solo incorpora un modo
            Movimientos grandes se hacen en la versión 0.1.0
            Debugging es 0.0.1"""
        self.image_widget = QLabel()
        self.image_widget.setText("BETA APP 1.2.0") 
        self.image_widget.setStyleSheet("QLabel { color : black; font: 36px; font-weight: bold; }")
    #Process of the cleaning 
    def clean_image(self):
        # Placeholder for cleaning image functionality
        QMessageBox.information(self, "Clean Image", "Clean Image functionality is not implemented yet.")
        return
    #Now working
    def process_detection(self):
        # Panel izquierdo con controles
        self.control_widget = QWidget()
        self.control_layout = QVBoxLayout()
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

        # Umbral máximo de área
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1000, 20000)
        self.max_area_spin.setValue(10500)
        self.parameters_layout.addRow("Area maxima detectable:", self.max_area_spin)

        # Umbral mínimo de área
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 100)
        self.min_area_spin.setValue(10)
        self.parameters_layout.addRow("Area minima detectable:", self.min_area_spin)

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

        # Panel derecho para mostrar imágenes
        self.image_widget = QLabel()
        self.image_widget.setFixedSize(800, 800)
        self.image_widget.setAlignment(Qt.AlignCenter)

        # Dividir la ventana en dos paneles
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.control_widget)
        self.splitter.addWidget(self.image_widget)
        self.setCentralWidget(self.splitter)
        QMessageBox.information(self, "Process Detection", "Process Detection functionality is usable.")
    #Process of the graphing
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

        # Actualizar label según el mod o
        if len(self.image_paths) > 1:
            # Modo de procesamiento en lote
            text = f"Procesando lote... {progress}%\nTiempo transcurrido: {elapsed_str}\nRestante: {remaining_str}"
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
            'max_area_threshold': self.max_area_spin.value(),
            'min_area_threshold': self.min_area_spin.value(),
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
            QApplication.processEvents()  # Actualizar la interfaz
                    
            # Configurar y lanzar el hilo
            self.get_parameters()
            self.processor = ImageProcessor([self.current_image_path], self.params)
            self.processor.image_processed.connect(self.on_image_processed)
            self.processor.progress_updated.connect(self.progress_bar.setValue)
            self.processor.batch_finished.connect(self.on_batch_finished) 
            self.processor.progress_updated.connect(self.update_progress)
            self.processor.error_occurred.connect(self.show_error)
            
            self.processor.start()

#ERROR HERE 


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
        self.processor.start()
        QApplication.processEvents()  # Actualizar la interfaz

    def on_image_processed(self, result_image, valid_count):
        """ Actualizar la GUI con los resultados de una imagen """
        self.timer.stop()  # 🔹 Detener temporizador
        self.update_elapsed_time() 
        self.display_result(result_image)
        QMessageBox.information(self, "Éxito", f"Burbujas detectadas: {valid_count}")

    def on_batch_finished(self, results_df):
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
        
        


if __name__ == "__main__":


    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())