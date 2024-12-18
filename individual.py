import cv2
import glfw
from OpenGL.GL import *
import sys
import numpy as np

# Variables globales
window = None
x_opengl = 0.0  # Posición X del cuadrado
y_opengl = 0.0  # Posición Y del cuadrado
scale = 0.1  # Escala del cuadrado
rotation_angle = 1  # Ángulo de rotación del cuadrado
prev_x = 0  # Posición X del contorno anterior (para calcular el movimiento)
prev_y = 0  # Posición Y del contorno anterior (para calcular el movimiento)

# Capturar video desde la cámara
video = cv2.VideoCapture(0)

def init():
    # Configuración inicial de OpenGL
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Color de fondo (negro)
    
    # Configuración de proyección ortogonal para 2D
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)

    # Cambiar a la matriz de modelo
    glMatrixMode(GL_MODELVIEW)

def draw_square(x, y, scale, angle):
    glPushMatrix()
    glTranslatef(x, y, 0.0)  # Posición del cuadrado
    glScalef(scale, scale, 5.0)  # Escalar el cuadrado
    glRotatef(angle, 0.0, 0.0, 1.0)  # Rotar el cuadrado en el eje Z

    glBegin(GL_QUADS)
    glColor3f(10.0, 5.0, 3.0)  # Amarillo
    glVertex2f(-0.5, -0.5)
    glVertex2f(0.5, -0.5)
    glVertex2f(0.5, 0.5)
    glVertex2f(-0.5, 0.5)
    glEnd()
    glPopMatrix()

def process_frame():
    global x_opengl, y_opengl, scale, rotation_angle, prev_x, prev_y

    ret, frame = video.read()
    if not ret:
        return

    # Convertir el frame a escala de grises
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar desenfoque para suavizar
    frame_blur = cv2.GaussianBlur(frame_gray, (15, 15), 0)

    # Detectar los contornos
    _, thresh = cv2.threshold(frame_blur, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:  # Si hay contornos
        # Encontrar el contorno más grande
        max_contour = max(contours, key=cv2.contourArea)

        # Calcular el rectángulo delimitador
        x, y, w, h = cv2.boundingRect(max_contour)

        # Convertir coordenadas a OpenGL (centrado en la pantalla)
        x_opengl = (x + w / 2 - frame.shape[1] / 2) / (frame.shape[1] / 2)
        y_opengl = -(y + h / 2 - frame.shape[0] / 2) / (frame.shape[0] / 2)

        # Ajustar escala basada en el tamaño del rectángulo
        scale = w / 300.0

        # Determinar el sentido del movimiento (hacia la derecha o izquierda)
        if prev_x != 0:
            delta_x = x - prev_x  # Diferencia en la posición X

            # Rotar dependiendo de la dirección del movimiento
            if delta_x > 0:  # Movimiento hacia la derecha
                rotation_angle += 5  # Incrementar el ángulo de rotación
            elif delta_x < 0:  # Movimiento hacia la izquierda
                rotation_angle -= 5  # Disminuir el ángulo de rotación

        # Actualizar las posiciones previas
        prev_x, prev_y = x, y

def main():
    global window

    # Inicializar GLFW
    if not glfw.init():
        sys.exit()

    # Crear ventana de GLFW
    width, height = 640, 480
    window = glfw.create_window(width, height, "Cuadrado 2D con Control por Mano", None, None)
    if not window:
        glfw.terminate()
        sys.exit()

    # Configurar el contexto de OpenGL en la ventana
    glfw.make_context_current(window)

    # Configuración de viewport y OpenGL
    glViewport(0, 0, width, height)
    init()

    # Bucle principal
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)  # Limpiar pantalla
        glLoadIdentity()

        # Procesar el frame para actualizar la posición y escala
        process_frame()

        # Dibujar el cuadrado con la rotación calculada
        draw_square(x_opengl, y_opengl, scale, rotation_angle)

        glfw.swap_buffers(window)  # Intercambiar buffers para animación suave
        glfw.poll_events()

    glfw.terminate()  # Cerrar GLFW al salir
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()