import numpy as np
import cv2
import json
import os

# --- 1. Configuración de parámetros ---
min_area_threshold = 150       
max_area_threshold = 15000    
min_aspect_ratio = 1       
min_rectangularity = 0.05     
min_brightness = 50

# PON AQUÍ LA RUTA EXACTA DE TU IMAGEN TIFF
image_filename = "Images\YZ25.tif" 

# Validar que la imagen existe antes de continuar
if not os.path.exists(image_filename):
    print(f"ERROR: No se encuentra el archivo '{image_filename}'. Revisa la ruta.")
    exit()

# --- 2. Carga y Normalización del TIFF ---
# cv2.IMREAD_UNCHANGED fuerza a OpenCV a leer la profundidad de bits real del TIFF
image = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)

# Si el TIFF es de 16-bits (pantalla negra), lo normalizamos a 8-bits (0-255)
if image.dtype == np.uint16:
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Convertir a gris para el procesamiento
image_gray = image.copy() if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h_img, w_img = image_gray.shape

# --- NUEVO: CREAR UNA COPIA PNG PARA LABELME ---
# LabelMe tiene muchos problemas con los TIFF, así que le daremos un PNG impecable
export_image_name = image_filename.lower().replace(".tif", ".png").replace(".tiff", ".png")
cv2.imwrite(export_image_name, image) # Guardamos la imagen visible

# El JSON tendrá el mismo nombre pero con extensión .json
json_filename = export_image_name.replace(".png", ".json")


# --- 3. Procesamiento Morfológico ---
kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
tophat = cv2.morphologyEx(image_gray, cv2.MORPH_TOPHAT, kernel_tophat)
_, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean, iterations=1)
contours, _ = cv2.findContours(clean_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- 4. Extracción de Coordenadas ---
shapes = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    
    if min_area_threshold <= area <= max_area_threshold:
        mask = np.zeros(image_gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        mean_val = cv2.mean(image_gray, mask=mask)[0]
        
        if mean_val < min_brightness:
            continue

        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        if w == 0 or h == 0: continue
        
        aspect_ratio = max(w, h) / min(w, h)
        rectangularity = area / (w * h)
        
        if aspect_ratio >= min_aspect_ratio and rectangularity >= min_rectangularity:
            box = cv2.boxPoints(rect)
            points_list = box.tolist() 
            
            shapes.append({
                "label": "fibra",
                "points": points_list,
                "group_id": None,
                "description": "",
                "shape_type": "polygon", 
                "flags": {}
            })

# --- 5. Exportar a JSON para LabelMe ---
labelme_data = {
    "version": "5.2.1",
    "flags": {},
    "shapes": shapes,
    "imagePath": os.path.basename(export_image_name), # Solo el nombre del archivo, sin la ruta completa
    "imageData": None,
    "imageHeight": h_img,
    "imageWidth": w_img
}

with open(json_filename, "w") as f:
    json.dump(labelme_data, f, indent=2)

# --- REPORTE DE RUTAS ---
print("-" * 50)
print(f"¡Exportación exitosa! Se encontraron {len(shapes)} fibras.")
print(f"Imagen convertida guardada en:\n-> {os.path.abspath(export_image_name)}")
print(f"Archivo JSON guardado en:\n-> {os.path.abspath(json_filename)}")
print("-" * 50)
print("Abre esa imagen .png en LabelMe y los cuadros cargarán automáticamente.")