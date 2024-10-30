import cv2 as cv
import numpy as np
import time

# Cargar la imagen en escala de grises
img = cv.imread('black.png', 0)

# Obtener el tamaño original de la imagen
x, y = img.shape

# Crear una nueva imagen para el escalado al doble
scaled_img = np.zeros((x * 2, y * 2), dtype=np.uint8)

# Escalar la imagen al doble utilizando ciclos for
for i in range(x * 2):
    for j in range(y * 2):
        orig_x = i // 2
        orig_y = j // 2
        scaled_img[i, j] = img[orig_x, orig_y]

# Crear los kernels (matrices) de convolución
# Kernel 1D para las direcciones horizontal y vertical
kernel_horizontal = np.array([[1, 2, 1]])
kernel_vertical = np.array([[1], [2], [1]])

# Kernel 2D para la segunda convolución
kernel_2d = np.array([
    [1, 2, 1],
    [2, 0, 2],
    [1, 2, 1]
])

# Aplicar convolución con el kernel 1D en ambas direcciones
start_time = time.time()
convoluted_img_1d = cv.filter2D(scaled_img, -1, kernel_horizontal)  # Filtro horizontal
convoluted_img_1d = cv.filter2D(convoluted_img_1d, -1, kernel_vertical)  # Filtro vertical
time_1d = time.time() - start_time

# Aplicar convolución con el kernel 2D
start_time = time.time()
convoluted_img_2d = cv.filter2D(scaled_img, -1, kernel_2d)
time_2d = time.time() - start_time

# Mostrar resultados y tiempos de ejecución
print(f"Tiempo de convolución con vectores [1, 2, 1]: {time_1d:.4f} segundos")
print(f"Tiempo de convolución con matriz [[1, 2, 1], [2, 0, 2], [1, 2, 1]]: {time_2d:.4f} segundos")

# Mostrar las imágenes originales y con convolución
cv.imshow('Imagen Original', img)
cv.imshow('Imagen Escalada x2', scaled_img)
cv.imshow('Imagen Convolución 1D [1, 2, 1]', convoluted_img_1d)
cv.imshow('Imagen Convolución 2D [[1, 2, 1], [2, 0, 2], [1, 2, 1]]', convoluted_img_2d)
cv.waitKey(0)
cv.destroyAllWindows()
