import cv2 as cv
import numpy as np

# Cargar la imagen en escala de grises
img = cv.imread('bob.png', 0)

# Obtener el tamaño original de la imagen
x, y = img.shape

# Crear una nueva imagen para el escalado al doble
scaled_img = np.zeros((x * 2, y * 2), dtype=np.uint8)

# Escalar la imagen al doble utilizando ciclos for
for i in range(x * 2):
    for j in range(y * 2):
        # Encontrar los índices correspondientes en la imagen original
        orig_x = i // 2
        orig_y = j // 2
        # Copiar el valor del píxel de la imagen original
        scaled_img[i, j] = img[orig_x, orig_y]

# Crear una nueva imagen para almacenar la versión suavizada
smoothed_img = np.zeros_like(scaled_img)

# Aplicar suavizado usando un filtro de promedio de 3x3
for i in range(1, scaled_img.shape[0] - 1):
    for j in range(1, scaled_img.shape[1] - 1):
        # Tomar el promedio de un bloque de 3x3 alrededor del píxel central
        neighborhood = scaled_img[i-1:i+2, j-1:j+2]
        smoothed_img[i, j] = np.mean(neighborhood)

# Mostrar la imagen original, la escalada y la suavizada
cv.imshow('Imagen Original', img)
cv.imshow('Imagen Escalada x2', scaled_img)
cv.imshow('Imagen Suavizada', smoothed_img)
cv.waitKey(0)
cv.destroyAllWindows()
