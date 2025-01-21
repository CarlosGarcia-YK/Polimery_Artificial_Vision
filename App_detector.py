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
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog, QSlider, QHBoxLayout,
    QVBoxLayout, QSplitter, QListWidget, QFormLayout, QSpinBox, QMessageBox, QProgressBar,QAction
)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage


#QThread para procesar las imágenes
class ImageProcessor(QThread):
    # Señales para comunicación con la GUI
    progress_updated = pyqtSignal(int)             # Progreso del procesamiento
    image_processed = pyqtSignal(np.ndarray, int)  # Imagen resultante y contador
    batch_finished = pyqtSignal(pd.DataFrame)      # Resultados finales del lote
    error_occurred = pyqtSignal(str)               # Mensajes de error

    def __init__(self, image_paths, params):
        super().__init__()
        self.image_paths = image_paths
        self.params = params
        self.is_running = True  # Bandera para controlar la ejecución

    def run(self):
        """ Método principal que se ejecuta en el hilo """
        try:
            all_batch_data = []
            total_images = len(self.image_paths)
            for idx, img_path in enumerate(self.image_paths):
                if not self.is_running:
                    break  # Detener si se solicita
                result, valid_count, _, batch_data = self.process_single_image(img_path, self.params)
                if result is not None:
                    all_batch_data.extend(batch_data)
                    self.image_processed.emit(result, valid_count)  # Enviar imagen procesada
                self.progress_updated.emit(int((idx + 1) / total_images * 100))
            self.batch_finished.emit(pd.DataFrame(all_batch_data))
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")

    def stop(self):
        """ Detener el procesamiento """
        self.is_running = False
    #The process for the images
    def process_single_image(self, image_path, params):
        max_area_threshold = params.get('max_area_threshold', 10500)
        min_area_threshold = params.get('min_area_threshold', 10)
        kernel_size = params.get('kernel_size', 3)
        morph_iterations = params.get('morph_iterations', 2)
        min_distance_peak = params.get('min_distance_peak', 8)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None, None, None, None
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        height_img, width_img = image.shape[:2]

        # 1. Suavizar la imagen
        blur = cv2.GaussianBlur(image, (3, 3), 0)

        # 2. Umbral Otsu
        _, binary = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)

        # 3. Limpiar por morfología
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)

        # 4. Transformada de distancia
        dist_transform = cv2.distanceTransform(binary_opened, cv2.DIST_L2, 5)

        # 5. Picos locales
        coordinates = peak_local_max(dist_transform, min_distance=min_distance_peak, threshold_abs=0.5)
        local_max = np.zeros(dist_transform.shape, dtype=bool)
        local_max[tuple(coordinates.T)] = True

        # Etiquetar burbujas
        markers, _ = ndimage.label(local_max)

        # 6. Identificación de burbujas
        labels = watershed(-dist_transform, markers, mask=binary_opened)

        # 7. Resultado
        num_labels = labels.max()
        colored_result = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        valid_count = 0
        batch_data = []

        scene_coordinate_system = 0.00
        height_pixels, width_pixels = image.shape[:2]
        escala_mm_px = 4.5 / 208  # Ajusta según corresponda

        # Proceso para calcular porcentaje no pintado
        total_pixels = binary_opened.size
        painted_pixels = np.count_nonzero(binary_opened)
        unpainted_pixels = total_pixels - painted_pixels
        unpainted_percentage = (unpainted_pixels / total_pixels) * 100
        unpainted_percentage_adjusted = max(0, unpainted_percentage - 15.35)

        pre_value = 0
        for lbl in range(1, num_labels + 1):
            mask = (labels == lbl).astype(np.uint8)
            area_px = cv2.countNonZero(mask)
            x, y, w, h = cv2.boundingRect(mask)
            aspect_ratio = w / h if h > 0 else 0

            # Filtro preliminar de burbujas válidas
            if 0.79 <= aspect_ratio <= 2.5 and min_area_threshold <= area_px <= max_area_threshold:
                pre_value += 1

        # Condiciones según el porcentaje no pintado ajustado
        recalculate = False
        if unpainted_percentage_adjusted > 25 and pre_value < 500:
            recalculate = True
        elif unpainted_percentage_adjusted < 6.0 or pre_value < 100:
            print(f"Unpainted percentage ({unpainted_percentage_adjusted}%) is too low or too high. Ignoring results...")
            return None, None, None, None
        else:
            print(f"Unpainted percentage ({unpainted_percentage_adjusted}%) is acceptable. Proceeding with results...")

        # Proceso para identificar burbujas
        for lbl in range(1, labels.max() + 1):
            mask = (labels == lbl).astype(np.uint8)
            area_px = cv2.countNonZero(mask)
            bubble_coordinates = coordinates[np.where(mask[tuple(coordinates.T)] == 1)]
            x, y, w, h = cv2.boundingRect(mask)
            aspect_ratio = w / h if h > 0 else 0

            if 0.7 <= aspect_ratio < 2.0 and min_area_threshold <= area_px <= max_area_threshold:
                # Calcular perímetro y otras métricas
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    perimeter_px = cv2.arcLength(contours[0], True)
                else:
                    perimeter_px = 0.0

                diametro_px = math.sqrt((4 * area_px) / math.pi)
                area_mm2 = area_px * (escala_mm_px ** 2)
                perimetro_mm = perimeter_px * escala_mm_px
                diametro_mm = diametro_px * escala_mm_px

                # Asignar color basado en el área
                color = get_color_by_increment(area_px, increment=150)
                colored_result[labels == lbl] = color

                class_id = classify_by_size(area_px, increase=150)

                # Guardar datos
                id_value = str(valid_count)
                batch_data.append({
                    'Imagen_idobject': image_name + '_' + id_value,
                    'Scene_coordinate_system': scene_coordinate_system,
                    'Area_px': area_px,
                    'Area_mm2': area_mm2,
                    'Perimetro_px': perimeter_px,
                    'Perimetro_mm': perimetro_mm,
                    'Diametro_px': diametro_px,
                    'Diametro_mm': diametro_mm,
                    'Escala_mm_por_px': escala_mm_px,
                    'Porcentaje_no_pintado': unpainted_percentage_adjusted,
                    'Coordenadas': bubble_coordinates.tolist(),
                    'Class_id': class_id,
                    'aspect_ratio': aspect_ratio
                })
                valid_count += 1

        if valid_count < 150:
            print(f"Burbujas detectadas ({valid_count}) es menor que 150. Ignorando resultados...")
            return None, None, None, None

        return colored_result, valid_count, unpainted_percentage_adjusted, batch_data

#Funciones unicas no se mueven
def get_color_by_increment(area, increment=150):
    # Determine the class (range) of the area
    class_id = area // increment  # Integer division
    random.seed(class_id)  # Ensure consistent color for the same class
    return (
        random.randint(0, 255), 
        random.randint(0, 255), 
        random.randint(0, 255)
    ) 
def classify_by_size(area,increase=150):
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
            if not self.current_image_path:
                QMessageBox.warning(self, "Error", "Selecciona una imagen primero.")
                return
            
            # Detener cualquier procesamiento previo
            if self.processor and self.processor.isRunning():
                self.processor.stop()
            
            # Configurar y lanzar el hilo
            self.get_parameters()
            self.processor = ImageProcessor([self.current_image_path], self.params)
            self.processor.image_processed.connect(self.on_image_processed)
            self.processor.progress_updated.connect(self.progress_bar.setValue)
            self.processor.error_occurred.connect(self.show_error)
            self.processor.start()

#ERROR HERE 


    def process_all_images(self):
        """ Procesar todas las imágenes en lote """
        if not self.image_paths:
            QMessageBox.warning(self, "Error", "No hay imágenes cargadas.")
            return
        
        # Detener cualquier procesamiento previo
        if self.processor and self.processor.isRunning():
            self.processor.stop()
        
        # Configurar y lanzar el hilo
        self.get_parameters()
        self.processor = ImageProcessor(self.image_paths, self.params)
        self.processor.progress_updated.connect(self.progress_bar.setValue)
        self.processor.batch_finished.connect(self.on_batch_finished)
        self.processor.error_occurred.connect(self.show_error)
        self.processor.start()

    def on_image_processed(self, result_image, valid_count):
        """ Actualizar la GUI con los resultados de una imagen """
        self.display_result(result_image)
        QMessageBox.information(self, "Éxito", f"Burbujas detectadas: {valid_count}")

    def on_batch_finished(self, results_df):
        """ Manejar los resultados del lote """
        self.download_button = QPushButton("Descargar CSV", self)
        self.download_button.clicked.connect(lambda: self.download_csv(results_df))
        self.control_layout.addWidget(self.download_button)

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


if __name__ == "__main__":


    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

