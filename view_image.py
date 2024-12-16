import cv2
import numpy as np

# Función que se llama cuando se hace clic en la imagen
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Muestra las coordenadas en la consola
        print(f"Coordenadas: ({x}, {y})")
        # Muestra las coordenadas en la imagen
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"({x}, {y})", (x, y), font, 0.5, (255, 0, 0), 2)
        cv2.imshow('image', img)

# Carga la imagen
img = cv2.imread(r'PU122-XY\PU122-XY\xy010.jpg')

# Redimensiona la imagen para que se ajuste a una ventana más pequeña
img_resized = cv2.resize(img, (800, 600))  # Ajusta el tamaño según tus necesidades

# Muestra la imagen redimensionada en una ventana
cv2.imshow('image', img_resized)

# Configura la función de clic del mouse
cv2.setMouseCallback('image', click_event)

# Espera hasta que se presione una tecla
cv2.waitKey(0)

# Cierra todas las ventanas
cv2.destroyAllWindows()
