import cv2 as cv
import numpy as np
import math

# Cargar la imagen
img = cv.imread('bob.png', 0)

# Obtener el tamaño de la imagen
x, y = img.shape

# Crear una imagen vacía para almacenar el resultado
result_img = np.zeros((int(x * 1/5), int(y * 1/5)), dtype=np.uint8)

# Definir el ángulo de rotación (en grados) y convertirlo a radianes
angle = 60
theta = math.radians(angle)

# Definir la traslación
dx, dy = 10, 10

# Definir la escala
scale = 1/5

# Rotar, trasladar y escalar la imagen
for i in range(int(x * scale)):
    for j in range(int(y * scale)):
        # Calcular las coordenadas originales
        orig_x = int(i / scale)
        orig_y = int(j / scale)

        # Rotar la imagen
        rotated_x = int((orig_x - x // 2) * math.cos(theta) - (orig_y - y // 2) * math.sin(theta) + x // 2)
        rotated_y = int((orig_x - x // 2) * math.sin(theta) + (orig_y - y // 2) * math.cos(theta) + y // 2)

        # Trasladar la imagen
        translated_x = rotated_x + dx
        translated_y = rotated_y + dy

        # Verificar si las coordenadas están dentro de la imagen
        if 0 <= translated_x < x and 0 <= translated_y < y:
            result_img[i, j] = img[translated_x, translated_y]

# Mostrar la imagen original y la imagen resultante
cv.imshow('Imagen Original', img)
cv.imshow('Imagen Rotada, Trasladada y Escalada', result_img)
cv.waitKey(0)
cv.destroyAllWindows()