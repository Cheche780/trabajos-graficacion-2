import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt
import cv2
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


def draw_house():
    """Dibuja una casa (base + techo)"""
    draw_cube()


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
    draw_circle(10, 0, (0.5, 0.5, 0.5))
    # Dibujar casas
    for i in range(8):
        angle = 2 * math.pi * i / 8
        x = 7 * math.cos(angle)
        z = 7 * math.sin(angle)
        glPushMatrix()
        glTranslatef(x, 0, z)
        glRotatef(-math.degrees(angle), 0, 1, 0)
        draw_house()
        glPopMatrix()
    glfw.swap_buffers(window)


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
            if abs(dx) > 2 or abs(dy) > 2:  # Detectar movimiento significativo
                if i == 0:
                    camera_yaw -= 1
                elif i == 2:
                    camera_yaw += 1
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
    cv2.destroyAllWindows()
    glfw.terminate()


if __name__ == "__main__":
    main()
