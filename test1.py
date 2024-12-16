import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import pandas as pd
import os
import glob
import math

#Pending 006 , 007, 008
all_images = glob.glob(r'XY_clean\Batch(11-20)*.JPG')
batch_size = 10
num_batches = math.ceil(len(all_images) / batch_size)
csv_filename = "Global_Results.csv"

for i in range(num_batches):
    batch_images = all_images[i*batch_size : (i+1)*batch_size]
    batch_data = []

    for image_path in batch_images:
        # Definir umbrales
        max_area_threshold = 10000
        min_area_threshold = 0

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # 1. Suavizar la imagen
        blur = cv2.GaussianBlur(image, (5, 5), 0)

        # 2. Umbral Otsu + invertir
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)

        # 3. Morfología para limpiar
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        binary_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=5)

        # 4. Dist transform
        dist_transform = cv2.distanceTransform(binary_opened, cv2.DIST_L2, 5)

        # 5. picos locales
        coordinates = peak_local_max(dist_transform, min_distance=10)
        local_max = np.zeros(dist_transform.shape, dtype=bool)
        local_max[tuple(coordinates.T)] = True

        # Etiquetas para watershed
        markers, _ = ndimage.label(local_max)

        # 6. Watershed
        labels = watershed(-dist_transform, markers, mask=binary_opened)

        # 7. Colorear resultado
        num_labels = labels.max()
        np.random.seed(42)
        colors = np.random.randint(0, 255, (num_labels+1, 3), dtype=np.uint8)

        colored_result = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        valid_count = 0

        # Si ya tienes escala_mm_px y scene_coordinate_system definidos, inclúyelos antes del bucle.
        # Por ejemplo:
        scene_coordinate_system = -19.94950
        width_pixels = image.shape[1]
        escala_mm_px = 4.5 / width_pixels

        for lbl in range(1, num_labels+1):
            mask = (labels == lbl).astype(np.uint8)
            area_px = cv2.countNonZero(mask)

            if min_area_threshold <= area_px <= max_area_threshold:
                # Calcular propiedades
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    perimeter_px = cv2.arcLength(contours[0], True)
                else:
                    perimeter_px = 0.0

                diametro_px = math.sqrt((4 * area_px) / math.pi)
                area_mm2 = area_px * (escala_mm_px**2)
                perimetro_mm = perimeter_px * escala_mm_px
                diametro_mm = diametro_px * escala_mm_px

                # Dibujar en colored_result
                colored_result[labels == lbl] = colors[lbl]

                # Agregar datos a batch_data
                batch_data.append({
                    'Imagen': image_name,
                    'Scene_coordinate_system': scene_coordinate_system,
                    'Area_px': area_px,
                    'Area_mm2': area_mm2,
                    'Perimetro_px': perimeter_px,
                    'Perimetro_mm': perimetro_mm,
                    'Diametro_px': diametro_px,
                    'Diametro_mm': diametro_mm,
                    'Escala_mm_por_px': escala_mm_px
                })

                valid_count += 1

        # Mostrar la imagen con las burbujas filtradas
        plt.figure(figsize=(12,12))
        plt.imshow(cv2.cvtColor(colored_result, cv2.COLOR_BGR2RGB))
        plt.title(f"Burbujas detectadas (filtradas por área): {valid_count}")
        plt.axis('off')
        plt.show()

    # Después de procesar el lote de imágenes:
    df_batch = pd.DataFrame(batch_data)
    print(df_batch.head())
    print(f"Se detectaron {len(df_batch)} burbujas en este lote.")
