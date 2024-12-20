# titulo 1

## proyectos chidotes de Graficacion 2024

## Jose Vidal Lopez Casimiro 
## No. Control: 22121271

###  arriba yo 
### mi apa
### y tupac 

#### actividad 1 

``` python 
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
```

### actividad 2 
``` python 
import cv2 as cv
import numpy as np
import math

img = cv.imread('bob.png', 0)


x, y = img.shape

result_img = np.zeros((int(x * 2), int(y * 2)), dtype=np.uint8)

angle1 = -30
theta1 = math.radians(angle1)

angle2 = 60
theta2 = math.radians(angle2)

scale = 2

for i in range(int(x * scale)):
    for j in range(int(y * scale)):

        orig_x = int(i / scale)
        orig_y = int(j / scale)


        rotated_x1 = int((orig_x - x // 2) * math.cos(theta1) - (orig_y - y // 2) * math.sin(theta1) + x // 2)
        rotated_y1 = int((orig_x - x // 2) * math.sin(theta1) + (orig_y - y // 2) * math.cos(theta1) + y // 2)

        rotated_x2 = int((rotated_x1 - x // 2) * math.cos(theta2) - (rotated_y1 - y // 2) * math.sin(theta2) + x // 2)
        rotated_y2 = int((rotated_x1 - x // 2) * math.sin(theta2) + (rotated_y1 - y // 2) * math.cos(theta2) + y // 2)

        if 0 <= rotated_x2 < x and 0 <= rotated_y2 < y:
            result_img[i, j] = img[rotated_x2, rotated_y2]


cv.imshow('Original Image', img)
cv.imshow('Rotated, Rotated, and Scaled Image', result_img)
cv.waitKey(0)
cv.destroyAllWindows()
```

### actividad 3 

```
import cv2 as cv
import numpy as np
import math

img = cv.imread('bob.png', 0)


x, y = img.shape


result_img = np.zeros((int(x * 2), int(y * 2)), dtype=np.uint8)


angle = 70
theta = math.radians(angle)


dx, dy = 20, 20


scale = 2


for i in range(int(x * scale)):
    for j in range(int(y * scale)):

        orig_x = int(i / scale)
        orig_y = int(j / scale)

        rotated_x = int((orig_x - x // 2) * math.cos(theta) - (orig_y - y // 2) * math.sin(theta) + x // 2)
        rotated_y = int((orig_x - x // 2) * math.sin(theta) + (orig_y - y // 2) * math.cos(theta) + y // 2)

        translated_x = rotated_x + dx
        translated_y = rotated_y + dy

        if 0 <= translated_x < x and 0 <= translated_y < y:
            result_img[i, j] = img[translated_x, translated_y]

cv.imshow('Original Image', img)
cv.imshow('Rotated, Translated, and Scaled Image', result_img)
cv.waitKey(0)
cv.destroyAllWindows()
```

#### actividad de animaciones parametricas.

![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)
![alt text](image-4.png)
![alt text](image-5.png)
![alt text](image-6.png)
![alt text](image-7.png)
![alt text](image-8.png)
![alt text](image-9.png)


### actividad Pelitita en Movimiento
```
import numpy as np
import cv2 as cv

# Iniciar la captura de video desde la cámara
cap = cv.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

# Obtener las dimensiones del video (ancho y alto de la cámara)
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Parámetros para el flujo óptico Lucas-Kanade
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Leer el primer frame de la cámara
ret, first_frame = cap.read()

if not ret:
    print("Error al capturar el primer frame")
    cap.release()
    exit()

first_frame = cv.flip(first_frame, 1)
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# Posición inicial de la pelotita (centrada en la imagen)
ball_pos = np.array([[frame_width // 2, frame_height // 2]], dtype=np.float32)
ball_pos = ball_pos[:, np.newaxis, :]

# Definir el recuadro azul 
margin = 50  # margen para ajustar el tamaño del rectángulo
rect_top_left = (margin, margin)
rect_bottom_right = (frame_width - margin, frame_height - margin)

while True:
    # Capturar el siguiente frame
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar frame")
        break

    frame = cv.flip(frame, 1)

    # Convertir el frame a escala de grises
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calcular el flujo óptico para mover la pelotita
    new_ball_pos, st, err = cv.calcOpticalFlowPyrLK(prev_gray, gray_frame, ball_pos, None, **lk_params)

    # Si se detecta el nuevo movimiento, actualizar la posición de la pelotita
    if new_ball_pos is not None:
        ball_pos = new_ball_pos

        # Obtener las coordenadas de la pelotita
        a, b = ball_pos.ravel()

        # Verificar si la pelotita se acerca a los bordes del rectángulo azul
        if (a <= rect_top_left[0] + 20 or a >= rect_bottom_right[0] - 20 or
                b <= rect_top_left[1] + 20 or b >= rect_bottom_right[1] - 20):
            # Si se acerca a los bordes, volver al centro
            ball_pos = np.array([[frame_width // 2, frame_height // 2]], dtype=np.float32)
            ball_pos = ball_pos[:, np.newaxis, :]

        # Dibujar la pelotita en su nueva posición
        a, b = ball_pos.ravel()
        frame = cv.circle(frame, (int(a), int(b)), 20, (0, 255, 0), -1)

    # Dibujar el recuadro azul (casi del tamaño de la pantalla)
    frame = cv.rectangle(frame, rect_top_left, rect_bottom_right, (255, 0, 0), 5)

    # Mostrar solo una ventana con la pelotita en movimiento
    cv.imshow('Pelota en movimiento', frame)

    # Actualizar el frame anterior para el siguiente cálculo
    prev_gray = gray_frame.copy()

    # Presionar 'Esc' para salir
    if cv.waitKey(30) & 0xFF == 27:
        break

# Liberar la captura y destruir todas las ventanas
cap.release()
cv.destroyAllWindows()
```

### Actividad Filtros como el Snap



### Actividad imagen convolucion 
```
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

```

### Actividad de convolucion entre separables y una matriz, comparacion en tiempos 
```
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
```

## actividad del triangulo de pascal
### chidota
```import pygame
from pygame.locals import *
from OpenGL.GL import *
import math

# Configuración inicial
width, height = 800, 600  # Tamaño de la ventana
filas = 10  # Número de filas del Triángulo de Pascal

def generar_triangulo_pascal(n):
    """Generar el Triángulo de Pascal como una lista de listas"""
    triangulo = []
    for i in range(n):
        fila = [1]
        if triangulo:
            ultima_fila = triangulo[-1]
            for j in range(len(ultima_fila) - 1):
                fila.append(ultima_fila[j] + ultima_fila[j + 1])
            fila.append(1)
        triangulo.append(fila)
    return triangulo

def inicializar_opengl():
    """Inicializar OpenGL sin GLUT"""
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Fondo negro
    glViewport(0, 0, width, height)  # Usar toda la ventana
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)  # Sistema de coordenadas ortográfico
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def dibujar_triangulo_pascal(triangulo):
    """Dibujar el Triángulo de Pascal"""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPointSize(5)  # Tamaño de los puntos

    base_x, base_y = 0.0, 0.9  # Coordenadas base
    espacio_x = 0.1  # Espaciado horizontal
    espacio_y = 0.1  # Espaciado vertical

    for i, fila in enumerate(triangulo):
        x_offset = -espacio_x * (len(fila) - 1) / 2  # Centramos cada fila
        for j, valor in enumerate(fila):
            glBegin(GL_POINTS)
            # Colores para pares e impares
            if valor % 2 == 0:
                glColor3f(0.0, 0.5, 1.0)  # Azul para pares
            else:
                glColor3f(1.0, 0.5, 0.0)  # Naranja para impares
            glVertex2f(base_x + x_offset, base_y - i * espacio_y)
            glEnd()
            x_offset += espacio_x  # Incrementar posición x

    pygame.display.flip()

def main():
    """Función principal"""
    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Triángulo de Pascal - OpenGL")

    inicializar_opengl()
    triangulo = generar_triangulo_pascal(filas)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False

        dibujar_triangulo_pascal(triangulo)

    pygame.quit()

if __name__ == "__main__":
    main()


```
## trabajo individual 
```
import numpy as np
import cv2 as cv
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

# Variables globales para OpenGL
window_width, window_height = 800, 600
rotation = [0, 0, 0]  
translation = [0, 0, -5]
scale = 1.0  


lkparm = dict(winSize=(15, 15), maxLevel=2,
              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))



def init_gl():
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, window_width / window_height, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)


def draw_prism():
    vertices = [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), 
        (4, 5), (5, 6), (6, 7), (7, 4),  
        (0, 4), (1, 5), (2, 6), (3, 7)   
    ]

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


def render():
    global translation, rotation, scale
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()


    glTranslatef(*translation)
    glScalef(scale, scale, scale)
    glRotatef(rotation[0], 1.0, 0.0, 0.0)
    glRotatef(rotation[1], 0.0, 1.0, 0.0)
    glRotatef(rotation[2], 0.0, 0.0, 1.0)

    draw_prism()
    glfw.swap_buffers(window)


def update_scale_based_on_direction():
    global scale, translation
    if translation[0] < 0: 
        scale += 0.01
        scale = min(scale, 3.0)  
    elif translation[0] > 0:  
        scale -= 0.01
        scale = max(scale, 0.5)  


def update_transformations(p1, p0):
    global translation, rotation
    delta = (p1 - p0).reshape(-1, 2)  # Garantiza forma (n, 2)
    mean_delta = np.mean(delta, axis=0)

    # Actualizar traslación y rotación
    translation[0] += mean_delta[0] * 0.01  # Movimiento horizontal
    translation[1] -= mean_delta[1] * 0.01  # Movimiento vertical
    rotation[1] += mean_delta[0] * 0.5  # Rotación sobre Y
    rotation[0] -= mean_delta[1] * 0.5  # Rotación sobre X

    # Actualizar escala en función de la dirección horizontal
    update_scale_based_on_direction()

# Bucle principal
def main():
    global window

    # Iniciar GLFW
    if not glfw.init():
        raise Exception("No se pudo inicializar GLFW")

    window = glfw.create_window(window_width, window_height, "Figura 3D con GLFW", None, None)
    if not window:
        glfw.terminate()
        raise Exception("No se pudo crear la ventana GLFW")

    glfw.make_context_current(window)
    init_gl()

    # Iniciar la captura de video
    cap = cv.VideoCapture(0)
    _, vframe = cap.read()
    vframe = cv.flip(vframe, 1)  # Invertir horizontalmente
    vgris = cv.cvtColor(vframe, cv.COLOR_BGR2GRAY)

    # Matriz inicial de puntos 7x7 más centrada
    rows, cols = vgris.shape
    step = 30
    p0 = np.array([[x, y] for y in range(rows // 2 - 90, rows // 2 + 90, step)
                   for x in range(cols // 2 - 90, cols // 2 + 90, step)])
    p0 = np.float32(p0[:, np.newaxis, :])

    while not glfw.window_should_close(window):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)  # Invertir horizontalmente
        fgris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calcular flujo óptico
        p1, st, err = cv.calcOpticalFlowPyrLK(vgris, fgris, p0, None, **lkparm)

        if p1 is not None:
            bp1 = p1[st == 1]
            bp0 = p0[st == 1]

            # Dibujar puntos cuadrados
            for nv in bp1:
                a, b = (int(x) for x in nv.ravel())
                top_left = (a - 3, b - 3)
                bottom_right = (a + 3, b + 3)
                frame = cv.rectangle(frame, top_left, bottom_right, (0, 0, 255), -1)

            # Actualizar transformaciones de OpenGL
            update_transformations(bp1, bp0)

        cv.imshow('Video', frame)
        render()
        vgris = fgris.copy()

        if cv.waitKey(1) & 0xFF == 27:
            break

        glfw.poll_events()

    glfw.terminate()
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

```

## PROYECTO FINAL
```
import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluNewQuadric, gluCylinder, gluSphere
from OpenGL.GLU import gluPerspective, gluLookAt
import cv2 
import cv2 as cv 
import numpy as np
import math
import sys



# Variables globales para el control de la cámara
camera_yaw = 0
camera_pitch = 60  # Ángulo inicial desde arriba (60 grados)
camera_distance = 20



# Dimensiones de la ventana y matriz de puntos
window_width, window_height = 800, 600
matrix_size = 3  # Matriz fija de 3x3


# Parámetros del flujo óptico
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Coordenadas de la matriz (se llenarán dinámicamente)
fixed_points = None

# Frame anterior para el cálculo del flujo óptico
prev_gray = None



def init_fixed_points(frame_width, frame_height):
    """Inicializa los puntos de la matriz para que estén centrados en la pantalla."""
    global fixed_points
    region_width = frame_width // (matrix_size + 1)  # Espaciado entre columnas
    region_height = frame_height // (matrix_size + 1)  # Espaciado entre filas

    fixed_points = np.array(
        [[(j + 1) * region_width, (i + 1) * region_height] for i in range(matrix_size) for j in range(matrix_size)],
        dtype=np.float32
    ).reshape(-1, 1, 2)
    

def init_opengl():
    """Configuración inicial de OpenGL"""
    glClearColor(0.5, 0.8, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(60, 1.0, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def draw_circle(radius, y, color):
    """Dibuja un círculo en el plano XZ"""
    glBegin(GL_TRIANGLE_FAN)
    glColor3f(*color)
    glVertex3f(0, y, 0)
    for i in range(101):
        angle = 2 * math.pi * i / 100
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        glVertex3f(x, y, z)
    glEnd()

def draw_cube():
    """Dibuja el cubo (base de la casa)"""
    glBegin(GL_QUADS)
    glColor3f(0.8, 0.5, 0.2)
    glVertex3f(-1, 0, 1)
    glVertex3f(1, 0, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 0, -1)
    glVertex3f(1, 0, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, 0, -1)
    glVertex3f(-1, 0, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 0, -1)
    glVertex3f(1, 0, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, 1, -1)
    glColor3f(0.9, 0.6, 0.3)
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)
    glColor3f(0.6, 0.4, 0.2)
    glVertex3f(-1, 0, -1)
    glVertex3f(1, 0, -1)
    glVertex3f(1, 0, 1)
    glVertex3f(-1, 0, 1)
    glEnd()

def draw_rectangular_prism(width, height, depth, color):
    """Dibuja un prisma rectangular con dimensiones y color especificados.

    Args:
        width: Ancho del prisma.
        height: Altura del prisma.
        depth: Profundidad del prisma.
        color: Tupla (R, G, B) que representa el color del prisma.
    """

    glColor3f(*color)  # Establece el color del prisma

    # Calcula los vértices del prisma
    x1 = -width/2
    x2 = width/2
    y1 = -height/2
    y2 = height/2
    z1 = -depth/2
    z2 = depth/2

    # Dibuja las caras del prisma utilizando GL_QUADS
    glBegin(GL_QUADS)
    # Cara frontal
    glVertex3f(x1, y1, z2)
    glVertex3f(x2, y1, z2)
    glVertex3f(x2, y2, z2)
    glVertex3f(x1, y2, z2)
    # Cara trasera
    glVertex3f(x1, y1, z1)
    glVertex3f(x2, y1, z1)
    glVertex3f(x2, y2, z1)
    glVertex3f(x1, y2, z1)
    # Cara izquierda
    glVertex3f(x1, y1, z1)
    glVertex3f(x1, y1, z2)
    glVertex3f(x1, y2, z2)
    glVertex3f(x1, y2, z1)
    # Cara derecha
    glVertex3f(x2, y1, z1)
    glVertex3f(x2, y1, z2)
    glVertex3f(x2, y2, z2)
    glVertex3f(x2, y2, z1)
    # Cara superior
    glVertex3f(x1, y2, z1)
    glVertex3f(x2, y2, z1)
    glVertex3f(x2, y2, z2)
    glVertex3f(x1, y2, z2)
    # Cara inferior
    glVertex3f(x1, y1, z1)
    glVertex3f(x2, y1, z1)
    glVertex3f(x2, y1, z2)
    glVertex3f(x1, y1, z2)
    glEnd()

def draw_house():
    """Dibuja una casa (base + techo)"""
    draw_cube()

def draw_sphere(radius, slices, stacks,color=(0, 1, 0)):
    """Dibuja una esfera usando coordenadas paramétricas."""
    glColor3f(*color)  # Establece el color de la esfera
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + i / stacks)
        z0 = radius * math.sin(lat0)
        zr0 = radius * math.cos(lat0)
        lat1 = math.pi * (-0.5 + (i + 1) / stacks)
        z1 = radius * math.sin(lat1)
        zr1 = radius * math.cos(lat1)
        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * (j / slices)
            x = math.cos(lng)
            y = math.sin(lng)
            glNormal3f(x * zr0, y * zr0, z0)
            glVertex3f(x * zr0, y * zr0, z0)
            glNormal3f(x * zr1, y * zr1, z1)
            glVertex3f(x * zr1,
                       y * zr1, z1)
        glEnd()

def draw_tronco(radius, height, slices, stacks, color):
    """Dibuja un tronco (cilindro) de un color sólido"""

    glColor3f(*color)  # Establece el color del tronco

    for i in range(stacks):
        theta0 = float(i) * 2 * math.pi / slices
        theta1 = float(i + 1) * 2 * math.pi / slices

        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            theta = float(j) * 2 * math.pi / slices
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)

            glVertex3f(x, y, i * height / stacks)
            glVertex3f(x, y, (i + 1) * height / stacks)
        glEnd()    

def draw_camera_marker(x, y, z):
    """Dibuja un marcador en la posición de la cámara (ajustado para no obstruir la vista)."""
    glPushMatrix()
    # Mueve el marcador un poco más abajo del punto de la cámara
    glTranslatef(x, y - 1, z)
    glColor3f(1.0, 0.0, 0.0)  # Rojo
    draw_sphere(0.01, 20, 20)  # Dibuja una esfera como marcador
    glPopMatrix()
    # Restablece el color para evitar que afecte otros objetos
    glColor3f(1.0, 1.0, 1.0)

def handle_optical_flow(camera_feed):
    """Calcula el flujo óptico y ajusta la cámara"""
    global prev_frame, camera_yaw, camera_pitch
    gray = cv2.cvtColor(camera_feed, cv2.COLOR_BGR2GRAY)
    if prev_frame is None:
        prev_frame = gray
        return
    prev_points = cv2.goodFeaturesToTrack(prev_frame, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    if prev_points is not None:
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, gray, prev_points, None, **lk_params)
        good_old = prev_points[status == 1]
        good_new = curr_points[status == 1]
        if len(good_old) > 0:
            motion = good_new - good_old
            avg_motion = np.mean(motion, axis=0)
            camera_yaw -= avg_motion[0] * 1.0
            camera_pitch += avg_motion[1] * 1.0
            camera_pitch = max(10, min(89, camera_pitch))
    prev_frame = gray

def draw_pyramid(x, y, z, size, color):
    """
    Dibuja una pirámide centrada en las coordenadas (x, y, z) con una base cuadrada de lado 'size' y un color especificado.

    Args:
        x: Coordenada x del centro de la base de la pirámide.
        y: Coordenada y del centro de la base de la pirámide.
        z: Coordenada z del centro de la base de la pirámide.
        size: Tamaño del lado de la base de la pirámide.
        color: Tupla (R, G, B) que representa el color de la pirámide.
    """

    glBegin(GL_TRIANGLES)
    glColor3f(*color)

    # Cara frontal
    glVertex3f(x - size/2, y, z + size/2)
    glVertex3f(x + size/2, y, z + size/2)
    glVertex3f(x, y + size, z)

    # Cara trasera
    glVertex3f(x - size/2, y, z - size/2)
    glVertex3f(x + size/2, y, z - size/2)
    glVertex3f(x, y + size, z)

    # Cara izquierda
    glVertex3f(x - size/2, y, z + size/2)
    glVertex3f(x - size/2, y, z - size/2)
    glVertex3f(x, y + size, z)

    # Cara derecha
    glVertex3f(x + size/2, y, z + size/2)
    glVertex3f(x + size/2, y, z - size/2)
    glVertex3f(x, y + size, z)

    glEnd()

def draw_rectangle(x, y, width, height, color):
    """
    Dibuja un rectángulo en las coordenadas especificadas.

    Args:
        x: Coordenada x de la esquina inferior izquierda del rectángulo.
        y: Coordenada y de la esquina inferior izquierda del rectángulo.
        width: Ancho del rectángulo.
        height: Alto del rectángulo.
        color: Tupla (R, G, B) que representa el color del rectángulo.
    """

    glBegin(GL_QUADS)
    glColor3f(*color)

    # Vertices del rectángulo
    glVertex3f(x, y, 0)  # Esquina inferior izquierda
    glVertex3f(x + width, y, 0)  # Esquina inferior derecha
    glVertex3f(x + width, y + height, 0)  # Esquina superior derecha
    glVertex3f(x, y + height, 0)  # Esquina superior izquierda

    glEnd()

def draw_street():
    # Dibujar calle
    glPushMatrix()
    glTranslatef(0, 0.1, 0)
    glRotatef(-math.degrees(80.1), 1, 0, 0)
    draw_rectangle(-20, 1.5, 40, 5, (0.3, 0.3, 0.3))
    glPopMatrix()
    # Dibujar linea central calle
    for i in range(8):
        x = 0 + i * 5
        z = 0
        glPushMatrix()
        glTranslatef(x, 0.11, z)
        glRotatef(-math.degrees(80.1), 1, 0, 0)
        draw_rectangle(-20, 3.8, 2, 0.3, (0.9, 0.9, 0.9))
        glPopMatrix()

def draw_arbol():
    # dibujar un arbol
    glPushMatrix()
    glTranslatef(1.5, 3, 0)
    glRotatef(-math.degrees(80.1), 1, 0, 0)
    draw_sphere(0.6, 32, 32)
    draw_tronco(0.18, 3.0, 32, 32, (0.5, 0.2, 0))
    glPopMatrix()

    glPushMatrix()
    glTranslatef(1, 2, 0)
    glRotatef(-math.degrees(80.1), 1, 0, 0)
    draw_sphere(0.6, 32, 32)
    glPopMatrix()
    
    glPushMatrix()
    glTranslatef(2, 2, 0)
    glRotatef(-math.degrees(80.1), 1, 0, 0)
    draw_sphere(0.6, 32, 32)
    glPopMatrix()
    
    glPushMatrix()
    glTranslatef(1.5, 2, 0.5)
    glRotatef(-math.degrees(80.1), 1, 0, 0)
    draw_sphere(0.6, 32, 32)
    glPopMatrix()
    
    glPushMatrix()
    glTranslatef(1.5, 2, -0.5)
    glRotatef(-math.degrees(80.1), 1, 0, 0)
    draw_sphere(0.6, 32, 32)
    glPopMatrix()

def draw_cube_walmart(x, y, z, width, height, depth, color):
    """Dibuja un cubo en una posición específica con dimensiones y color dados"""
    glPushMatrix()
    glTranslatef(x, y, z)
    glScalef(width, height, depth)
    glColor3f(*color)
    glBegin(GL_QUADS)

    # Frente
    glVertex3f(-0.5, -0.5, 0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)

    # Atrás
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(-0.5, 0.5, -0.5)

    # Izquierda
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, -0.5)

    # Derecha
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, -0.5)

    # Arriba
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)

    # Abajo
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(-0.5, -0.5, 0.5)

    glEnd()
    glPopMatrix()

def draw_text_on_cube(x, y, z, text, color):
    """Dibuja texto básico simulando caracteres como líneas sobre un cubo."""
    glColor3f(*color)
    for i, char in enumerate(text):
        glPushMatrix()
        glTranslatef(x + i * 0.3, y, z)  # Posicionar cada letra
        glScalef(0.1, 0.1, 0.1)
        draw_letter(char)
        glPopMatrix()

def draw_letter(char):
    """Dibuja letras básicas como líneas simuladas (solo para algunas letras)."""
    glBegin(GL_LINES)
    if char == 'U':
        glVertex3f(-0.5, -0.5, 0.0)
        glVertex3f(-0.5, 0.5, 0.0)
        glVertex3f(0.5, -0.5, 0.0)
        glVertex3f(0.5, 0.5, 0.0)
        glVertex3f(-0.5, -0.5, 0.0)
        glVertex3f(0.5, -0.5, 0.0)
    elif char == 'A':
        glVertex3f(-0.5, -0.5, 0.0)
        glVertex3f(0.0, 0.5, 0.0)
        glVertex3f(0.5, -0.5, 0.0)
        glVertex3f(0.0, 0.5, 0.0)
        glVertex3f(-0.25, 0.0, 0.0)
        glVertex3f(0.25, 0.0, 0.0)
    elif char == 'L':
        glVertex3f(-0.5, 0.5, 0.0)
        glVertex3f(-0.5, -0.5, 0.0)
        glVertex3f(-0.5, -0.5, 0.0)
        glVertex3f(0.5, -0.5, 0.0)
    elif char == 'M':
        glVertex3f(-0.5, -0.5, 0.0)
        glVertex3f(-0.5, 0.5, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(-0.5, 0.5, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.5, 0.5, 0.0)
        glVertex3f(0.5, 0.5, 0.0)
        glVertex3f(0.5, -0.5, 0.0)
    elif char == 'R':
        glVertex3f(-0.5, -0.5, 0.0)
        glVertex3f(-0.5, 0.5, 0.0)
        glVertex3f(-0.5, 0.5, 0.0)
        glVertex3f(0.0, 0.5, 0.0)
        glVertex3f(0.0, 0.5, 0.0)
        glVertex3f(0.5, 0.0, 0.0)
        glVertex3f(-0.5, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
    elif char == 'T':
        glVertex3f(-0.5, 0.5, 0.0)
        glVertex3f(0.5, 0.5, 0.0)
        glVertex3f(0.0, 0.5, 0.0)
        glVertex3f(0.0, -0.5, 0.0)
    glEnd()

def draw_supermarket():
    # Suelo
    draw_cube_walmart(0, -0.1, 0, 20, 0.1, 20, (0.2, 0.6, 0.2))

    # Edificio principal
    draw_cube_walmart(0, 1, 0, 6, 2, 4, (0.8, 0.2, 0.2))

    # Ventanas
    draw_cube_walmart(-2, 1, 2.01, 0.8, 0.8, 0.1, (0.6, 0.8, 1.0))
    draw_cube_walmart(0, 1, 2.01, 0.8, 0.8, 0.1, (0.6, 0.8, 1.0))
    draw_cube_walmart(2, 1, 2.01, 0.8, 0.8, 0.1, (0.6, 0.8, 1.0))

    # Letrero
    draw_cube_walmart(0, 2.6, 0, 4, 0.2, 0.2, (0.0, 0.0, 0.5))
    draw_text_on_cube(-1.5, 2.6, 0.3, "UALMART", (1.0, 1.0, 1.0))

# Función para dibujar el suelo (plano)
def draw_parking_floor():
    glPushMatrix()
    glColor3f(0.1, 0.1, 0.1)  # Color gris para el suelo
    glBegin(GL_QUADS)
    glVertex3f(-10, -0.1, -10)  # Esquina inferior izquierda
    glVertex3f(10, -0.1, -10)  # Esquina inferior derecha
    glVertex3f(10, -0.1, 10)  # Esquina superior derecha
    glVertex3f(-10, -0.1, 10)  # Esquina superior izquierda
    glEnd()
    glPopMatrix()


# Función para dibujar un espacio de estacionamiento
def draw_parking_space(x, y, width, depth):
    glPushMatrix()
    glTranslatef(x, 0.0, y)  # Posicionar cada espacio de estacionamiento
    glColor3f(0.8, 0.8, 0.8)  # Color gris claro para los espacios
    glBegin(GL_QUADS)
    glVertex3f(-width / 2, 0.0, -depth / 2)
    glVertex3f(width / 2, 0.0, -depth / 2)
    glVertex3f(width / 2, 0.0, depth / 2)
    glVertex3f(-width / 2, 0.0, depth / 2)
    glEnd()
    glPopMatrix()


# Función para dibujar las líneas divisorias del estacionamiento
def draw_parking_lines():
    glPushMatrix()
    glColor3f(10.0, 5.0, 10.0)  # Color blanco para las líneas divisorias

    # Dibujar líneas divisorias entre los espacios
    glBegin(GL_LINES)

    # Líneas verticales
    for x in range(-9, 10, 2):
        glVertex3f(x, 0.0, -10)
        glVertex3f(x, 0.0, 10)

    # Líneas horizontales
    for y in range(-9, 10, 2):
        glVertex3f(-10, 0.0, y)
        glVertex3f(10, 0.0, y)

    glEnd()
    glPopMatrix()

def draw_parking():
    # Dibujar el suelo y el estacionamiento
    draw_parking_floor()  # Dibujar el suelo
    draw_parking_lines()  # Dibujar las líneas divisorias

    # Dibujar los espacios de estacionamiento (puedes modificar el número de espacios)
    for x in range(-8, 10, 4):  # Espacios de estacionamiento en X
        for y in range(-8, 10, 4):  # Espacios de estacionamiento en Y
            draw_parking_space(x, y, 2, 4)

def draw_estatua():
    # Dibuja la base de la estatua
    glBegin(GL_QUADS)
    glColor3f(0.5, 0.5, 0.5)  # Color gris
    glVertex3f(-1, -1, 0)
    glVertex3f(1, -1, 0)
    glVertex3f(1, 1, 0)
    glVertex3f(-1, 1, 0)
    glEnd()

    # Dibuja el cuerpo de la estatua
    glBegin(GL_QUADS)
    glColor3f(0.7, 0.7, 0.7)  # Color gris claro
    glVertex3f(-0.5, -1, 0)
    glVertex3f(0.5, -1, 0)
    glVertex3f(0.5, 2, 0)
    glVertex3f(-0.5, 2, 0)
    glEnd()

    # Dibuja la cabeza de la estatua
    glBegin(GL_QUADS)
    glColor3f(0.9, 0.9, 0.9)  # Color gris claro
    glVertex3f(-0.2, 2, 0)
    glVertex3f(0.2, 2, 0)
    glVertex3f(0.2, 3, 0)
    glVertex3f(-0.2, 3, 0)
    glEnd()


def draw_hotdog_cart():
    # Base del carrito
    draw_cube_walmart(0, 0.5, 0, 4, 0.5, 2, (0.8, 0.2, 0.2))

    # Ruedas
    draw_cube_walmart(-1.5, 0.2, 1, 0.5, 0.5, 0.5, (0.2, 0.2, 0.2))
    draw_cube_walmart(1.5, 0.2, 1, 0.5, 0.5, 0.5, (0.2, 0.2, 0.2))
    draw_cube_walmart(-1.5, 0.2, -1, 0.5, 0.5, 0.5, (0.2, 0.2, 0.2))
    draw_cube_walmart(1.5, 0.2, -1, 0.5, 0.5, 0.5, (0.2, 0.2, 0.2))

    # Techado del carrito
    draw_cube_walmart(0, 1.5, 0, 4.2, 0.2, 2.2, (0.9, 0.9, 0.9))

    # Postes laterales
    draw_cube_walmart(-1.8, 1, 0.9, 0.1, 1.0, 0.1, (0.8, 0.8, 0.8))
    draw_cube_walmart(1.8, 1, 0.9, 0.1, 1.0, 0.1, (0.8, 0.8, 0.8))
    draw_cube_walmart(-1.8, 1, -0.9, 0.1, 1.0, 0.1, (0.8, 0.8, 0.8))
    draw_cube_walmart(1.8, 1, -0.9, 0.1, 1.0, 0.1, (0.8, 0.8, 0.8))

    # Parrilla del carrito
    draw_cube_walmart(0, 1.0, 0, 3.8, 0.1, 1.8, (0.5, 0.5, 0.5))

    # Manija trasera del carrito
    draw_cube_walmart(-2.2, 1.0, 0, 0.2, 0.5, 0.1, (0.3, 0.3, 0.3))
    draw_cube_walmart(-2.2, 1.25, 0, 0.2, 0.1, 1.2, (0.3, 0.3, 0.3))

    # Hot dogs sobre la parrilla
    draw_cube_walmart(-0.5, 1.1, 0.5, 0.8, 0.2, 0.2, (0.8, 0.4, 0.1))
    draw_cube_walmart(0.5, 1.1, 0.5, 0.8, 0.2, 0.2, (0.8, 0.4, 0.1))
    draw_cube_walmart(0, 1.1, -0.5, 0.8, 0.2, 0.2, (0.8, 0.4, 0.1))

def draw_semaforo():
    # dibujar semaforo
    glPushMatrix()
    glTranslatef(0, 3, 0)
    draw_rectangular_prism(0.5, 2, 0.5, (0, 0, 0))
    glPopMatrix()
    # luz 1
    glPushMatrix()
    glTranslatef(0.26, 3.8, 0)
    glRotatef(90, 0, 0, 1)
    draw_circle(0.18, 0, (0, 1, 0))
    glPopMatrix()
    # luz 2
    glPushMatrix()
    glTranslatef(0.26, 3.33, 0)
    glRotatef(90, 0, 0, 1)
    draw_circle(0.18, 0, (1, 1, 0))
    glPopMatrix()
    # luz 3
    glPushMatrix()
    glTranslatef(0.26, 2.8, 0)
    glRotatef(90, 0, 0, 1)
    draw_circle(0.18, 0, (1, 0, 0))
    glPopMatrix()
    # poste
    glPushMatrix()
    glTranslatef(-0.16, 2, 0)
    glRotatef(90, 1, 0, 0)
    draw_tronco(0.15, 2, 10, 10, (0, 0, 0))
    glPopMatrix()

def draw_carro():
    # Base del carro (cuerpo principal en amarillo)
    draw_cube_walmart(0, 0.5, 0, 6, 0.5, 3, (1.0, 1.0, 0.0))  # Color amarillo

    # Ruedas
    draw_cube_walmart(-2.5, 0.25, 1.5, 0.5, 0.5, 0.5, (0.1, 0.1, 0.1))  # Frontal izquierda
    draw_cube_walmart(2.5, 0.25, 1.5, 0.5, 0.5, 0.5, (0.1, 0.1, 0.1))   # Frontal derecha
    draw_cube_walmart(-2.5, 0.25, -1.5, 0.5, 0.5, 0.5, (0.1, 0.1, 0.1)) # Trasera izquierda
    draw_cube_walmart(2.5, 0.25, -1.5, 0.5, 0.5, 0.5, (0.1, 0.1, 0.1))  # Trasera derecha

    # Ventanas laterales
    draw_cube_walmart(0, 1.5, 1.4, 5.5, 0.4, 0.1, (0.5, 0.8, 1.0))  # Lateral derecha
    draw_cube_walmart(0, 1.5, -1.4, 5.5, 0.4, 0.1, (0.5, 0.8, 1.0)) # Lateral izquierda

    # Luces frontales
    draw_cube_walmart(-2, 0.7, 1.6, 0.4, 0.4, 0.1, (1, 1, 0))  # Izquierda
    draw_cube_walmart(2, 0.7, 1.6, 0.4, 0.4, 0.1, (1, 1, 0))   # Derecha

    # Luces traseras
    draw_cube_walmart(-2, 0.7, -1.6, 0.4, 0.4, 0.1, (1, 0, 0)) # Izquierda
    draw_cube_walmart(2, 0.7, -1.6, 0.4, 0.4, 0.1, (1, 0, 0))  # Derecha

def draw_stop_sign():
    # Color rojo para la señal
    glColor3f(1.0, 0.0, 0.0)  # Rojo
    glBegin(GL_POLYGON)
    for i in range(8):
        angle = math.radians(i * 45)  # 360/8 = 45 grados
        x = 1.0 * math.cos(angle)
        y = 1.0 * math.sin(angle)
        glVertex3f(x, y, 0)  # Vértices del octágono
    glEnd()

    # Opcional: dibujar un borde blanco alrededor de la señal
    glColor3f(1.0, 1.0, 1.0)  # Blanco
    glBegin(GL_LINE_LOOP)
    for i in range(8):
        angle = math.radians(i * 45)  # 360/8 = 45 grados
        x = 1.0 * math.cos(angle)
        y = 1.0 * math.sin(angle)
        glVertex3f(x, y, 0.01)  # Vértices del borde
    glEnd()

def draw_post():
    # Poste de la señal
    glColor3f(1, 1, 1)  # Color gris
    glBegin(GL_QUADS)
    glVertex3f(-0.05, -1, 0)  # Esquina inferior izquierda
    glVertex3f(0.05, -1, 0)   # Esquina inferior derecha
    glVertex3f(0.05, 1, 0)    # Esquina superior derecha
    glVertex3f(-0.05, 1, 0)   # Esquina superior izquierda
    glEnd()

def draw_stop():
    # dibujar senal de stop
    glPushMatrix()
    glTranslatef(-3, 5, 0)
    draw_stop_sign()
    glPopMatrix()
    glPushMatrix()
    glTranslatef(-3, 3, 0)
    draw_post()
    glPopMatrix() 
def draw_farola():
    # Factor de escala para reducir el tamaño a la mitad
    escala = 0.5

    # Dibujar el poste de la farola
    glPushMatrix()
    glColor3f(0.3, 0.3, 0.3)
    glRotatef(-90, 1, 0, 0)
    glTranslate(0,0,0)
    glScalef(escala, escala, escala)  # Aplicamos la escala
    gluCylinder(gluNewQuadric(), 0.2, 0.2, 10.0 * escala, 32, 32)
    glPopMatrix()

    # Dibujar el brazo de la farola
    glPushMatrix()
    glColor3f(0.3, 0.3, 0.3)
    glTranslate(0,-2,0)
    glTranslatef(0.0, 8.0 * escala, 0.0)
    glRotatef(90, 0, 1, 0)
    glScalef(escala, escala, escala)
    gluCylinder(gluNewQuadric(), 0.2, 0.2, 2.0 * escala, 32, 32)
    glPopMatrix()

    # Dibujar el foco (lámpara)
    glPushMatrix()
    glColor3f(1.0, 1.0, 0.0)
    glTranslate(-0.5,-2,0)
    glTranslatef(2.0 * escala, 8.0 * escala, 0.0)
    glScalef(escala, escala, escala)
    gluSphere(gluNewQuadric(), 0.5 * escala, 32, 32)
    glPopMatrix()

    # Dibujar la base de la farola
    glPushMatrix()
    glColor3f(0.2, 0.2, 0.2)
    glTranslatef(0.0, -0.5 * escala, 0.0)
    glRotatef(-90, 1, 0, 0)
    glScalef(escala, escala, escala)
    gluCylinder(gluNewQuadric(), 0.5 * escala, 0.7 * escala, 0.5 * escala, 32, 32)
    glPopMatrix()
       
def draw_scene():
    """Dibuja toda la escena"""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    # Configuración de la cámara
    camera_x = camera_distance * math.cos(math.radians(camera_yaw)) * math.cos(math.radians(camera_pitch))
    camera_z = camera_distance * math.sin(math.radians(camera_yaw)) * math.cos(math.radians(camera_pitch))
    camera_y = camera_distance * math.sin(math.radians(camera_pitch))
    gluLookAt(camera_x, camera_y, camera_z, 0, 0, 0, 0, 1, 0)
    # Dibujar el suelo
    draw_circle(30, 0, (0.55, 0.55, 0.55))
    draw_street()
    glPushMatrix()
    glTranslatef(0, 0, 0)
    glRotatef(-math.degrees(80.1), 0, 1, 0)
    draw_street()
    glPopMatrix()
    # Dibujar casas
    for i in range(8):
        if i == 4:
            continue
        else:
            angle = 0
            x = -15 + i * 5
            z = 0
            glPushMatrix()
            glTranslatef(x, 0, z)
            glRotatef(-math.degrees(angle), 0, 1, 0)
            draw_house()
            glPopMatrix()
    for i in range(8):
        if i == 4:
            continue
        else:
            angle = 0
            x = -15 + i * 5
            z = 8
            glPushMatrix()
            glTranslatef(x, 0, z)
            glRotatef(-math.degrees(angle), 0, 1, 0)
            draw_house()
            glPopMatrix()
    # dibujar techos
    for i in range(8):
        if i == 4:
            continue
        else:
            x=-15.5 + i*5
            glPushMatrix()
            glTranslatef(x, 0.5, 0)
            glRotatef(-math.degrees(angle), 0, 1, 0)
            draw_pyramid(0.5, 0.5, 0, 2, (0.1, 0.1, 0.1))
            glPopMatrix()
    for i in range(8):
        if i == 4:
            continue
        else:
            x=-15.5 + i*5
            z=8
            glPushMatrix()
            glTranslatef(x, 0.5, z)
            glRotatef(-math.degrees(angle), 0, 1, 0)
            draw_pyramid(0.5, 0.5, 0, 2, (0.2, 0.2, 0.2))
            glPopMatrix()
    for i in range(1):
        x=-3.6 + i*5
        glPushMatrix()
        glTranslatef(x, 0, 1)
        draw_arbol()
        glPopMatrix()

    # dibujar edificios
    glPushMatrix()
    glTranslatef(-0.5, 4.5, -5)
    draw_rectangular_prism(3,8,5,(0,0,0))
    glPopMatrix()

    # dibujar walmart
    glPushMatrix()
    glTranslatef(9, 0, 13)
    glRotatef(90, 0, 1, 0)
    draw_supermarket()
    glPopMatrix()
    # dibujar farolas
    glPushMatrix()
    glTranslatef(1.3, 0.01, 1)
    draw_farola()
    glPopMatrix()
    glPushMatrix()
    glTranslatef(6.8, 0.01, 1)
    glRotatef(-180, 0, 1, 0)
    draw_farola()
    glPopMatrix()
    # dibujar stacionamiento
    glPushMatrix()
    glTranslatef(-12,1,20)
    draw_parking()
    glPopMatrix()
    # dibujar farolas
    glPushMatrix()
    glTranslatef(1.3, 0.01, 7)
    draw_farola()
    glPopMatrix()
    glPushMatrix()
    glTranslatef(6.8, 0.01, 7)
    glRotatef(-180, 0, 1, 0)
    draw_farola()
    glPopMatrix()
    # dibujar stacionamiento
    glPushMatrix()
    glTranslatef(-12,1,20)
    draw_parking()
    glPopMatrix()

    # dibujar estatua
    glPushMatrix()
    glTranslatef(-8,0.5, 18)
    draw_estatua()
    glPopMatrix()
    # Dibujar carrito hotdogs
    glPushMatrix()
    glTranslatef(10,1,-5)
    draw_hotdog_cart()
    glPopMatrix()
    # dibujar semaforo
    glPushMatrix()
    glTranslatef(1,0,1.5)
    draw_semaforo()
    glPopMatrix()
    # dibujar carro
    glPushMatrix()
    glTranslatef(3, 1, 4)
    draw_carro()
    glPopMatrix()
    # Dibujar el marcador de la cámara
    draw_camera_marker(camera_x, camera_y, camera_z)
    glfw.swap_buffers(window)
    #dibujar stop
    glPushMatrix()
    glTranslatef ( 2, 3, 3)
    draw_stop()
    glPopMatrix()
    
    


def process_optical_flow(frame):
    global prev_gray, camera_yaw, camera_distance, camera_pitch

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        return

    # Calcular el flujo óptico
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, fixed_points, None, **lk_params)

    for i, (new, old) in enumerate(zip(new_points, fixed_points)):
        if status[i]:
            dx, dy = new.ravel() - old.ravel()
            if abs(dx) > 3 or abs(dy) > 3:  # Detectar movimiento significativo
                if i == 0:
                    camera_yaw -= 2
                elif i == 2:
                    camera_yaw += 2
                elif i == 6:
                    camera_distance = max(5, camera_distance - 0.5)
                elif i == 8:
                    camera_distance = min(30, camera_distance + 0.5)

    prev_gray = gray


def draw_matrix_on_camera(frame):
    """Dibuja la matriz de puntos en la ventana de la cámara"""
    for point in fixed_points:
        x, y = point.ravel()
        cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), -1)


def main():
    
    
    global window
    if not glfw.init():
        sys.exit()
    window = glfw.create_window(800, 600, "Proyecto final", None, None)
    if not window:
        glfw.terminate()
        sys.exit()
    glfw.make_context_current(window)
    glViewport(0, 0, 800, 600)
    init_opengl()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        sys.exit()

    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame de la cámara.")
        sys.exit()

    init_fixed_points(frame.shape[1], frame.shape[0])  # Inicializar matriz centrada

    if not glfw.init():
        sys.exit()

    window = glfw.create_window(window_width, window_height, "Escena 3D", None, None)
    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)
    glViewport(0, 0, window_width, window_height)
    init_opengl()

    while not glfw.window_should_close(window):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Voltear la imagen para una experiencia más intuitiva

        draw_matrix_on_camera(frame)  # Dibujar la matriz de puntos
        process_optical_flow(frame)  # Procesar flujo óptico

        cv2.imshow("Cámara", frame)  # Mostrar la ventana de la cámara
        draw_scene()  # Dibujar la escena 3D

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        glfw.poll_events()
    cap.release()
    glfw.terminate()

if __name__ == "__main__":
    main()

```

