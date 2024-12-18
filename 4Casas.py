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

# Variables de flujo óptico
prev_frame = None
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parámetros de la matriz de puntos
matrix_size = (10, 10)  # Tamaño de la matriz de puntos (filas, columnas)
matrix_spacing = 30  # Espaciado entre puntos en píxeles
p0 = None  # Puntos iniciales de la matriz

# Rotación del objeto
object_yaw = 0
object_pitch = 0

def init_opengl():
    """Configuración inicial de OpenGL"""
    glClearColor(0.5, 0.8, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(60, 1.0, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def draw_cube():
    """Dibuja un cubo en el origen"""
    glBegin(GL_QUADS)
    glColor3f(0.8, 0.5, 0.2)
    # Frente
    glVertex3f(-1, -1,  1)
    glVertex3f( 1, -1,  1)
    glVertex3f( 1,  1,  1)
    glVertex3f(-1,  1,  1)
    # Atrás
    glVertex3f(-1, -1, -1)
    glVertex3f( 1, -1, -1)
    glVertex3f( 1,  1, -1)
    glVertex3f(-1,  1, -1)
    # Izquierda
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, -1,  1)
    glVertex3f(-1,  1,  1)
    glVertex3f(-1,  1, -1)
    # Derecha
    glVertex3f( 1, -1, -1)
    glVertex3f( 1, -1,  1)
    glVertex3f( 1,  1,  1)
    glVertex3f( 1,  1, -1)
    # Arriba
    glColor3f(0.9, 0.6, 0.3)
    glVertex3f(-1,  1, -1)
    glVertex3f( 1,  1, -1)
    glVertex3f( 1,  1,  1)
    glVertex3f(-1,  1,  1)
    # Abajo
    glColor3f(0.6, 0.4, 0.2)
    glVertex3f(-1, -1, -1)
    glVertex3f( 1, -1, -1)
    glVertex3f( 1, -1,  1)
    glVertex3f(-1, -1,  1)
    glEnd()

def draw_scene():
    """Dibuja toda la escena"""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    # Configuración de la cámara
    camera_x = camera_distance * math.cos(math.radians(camera_yaw)) * math.cos(math.radians(camera_pitch))
    camera_z = camera_distance * math.sin(math.radians(camera_yaw)) * math.cos(math.radians(camera_pitch))
    camera_y = camera_distance * math.sin(math.radians(camera_pitch))
    gluLookAt(camera_x, camera_y, camera_z, 0, 0, 0, 0, 1, 0)
    # Dibujar el cubo en el centro
    glPushMatrix()
    glRotatef(object_pitch, 1, 0, 0)
    glRotatef(object_yaw, 0, 1, 0)
    draw_cube()
    glPopMatrix()
    glfw.swap_buffers(window)

def initialize_matrix(frame):
    """Inicializa la matriz de puntos para el flujo óptico"""
    global p0
    h, w = frame.shape[:2]
    x_start = w // 2 - (matrix_size[1] - 1) * matrix_spacing // 2
    y_start = h // 2 - (matrix_size[0] - 1) * matrix_spacing // 2
    points = [[(x_start + j * matrix_spacing, y_start + i * matrix_spacing) for j in range(matrix_size[1])] for i in range(matrix_size[0])]
    p0 = np.float32(points).reshape(-1, 1, 2)

def handle_optical_flow(camera_feed):
    """Calcula el flujo óptico y ajusta la rotación y el zoom"""
    global prev_frame, camera_distance, object_yaw, object_pitch, p0
    gray = cv2.cvtColor(camera_feed, cv2.COLOR_BGR2GRAY)
    if prev_frame is None:
        prev_frame = gray
        initialize_matrix(gray)
        return
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, gray, p0, None, **lk_params)
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        motion = good_new - good_old
        avg_motion = np.mean(motion, axis=0) if len(motion) > 0 else [0, 0]

        # Ajustar rotación del objeto
        object_yaw -= avg_motion[0] * 0.5
        object_pitch += avg_motion[1] * 0.5
        object_pitch = max(-89, min(89, object_pitch))

        # Ajustar zoom según densidad de puntos
        active_points = len(good_new)
        camera_distance = 20 - (active_points / (matrix_size[0] * matrix_size[1])) * 10
        camera_distance = max(5, min(20, camera_distance))

        p0 = good_new.reshape(-1, 1, 2)
    prev_frame = gray

def draw_matrix_on_frame(frame):
    """Dibuja la matriz de puntos sobre el frame de la cámara"""
    if p0 is not None:
        for point in p0:
            x, y = point.ravel()
            cv2.rectangle(frame, (int(x) - 2, int(y) - 2), (int(x) + 2, int(y) + 2), (0, 0, 255), -1)
    return frame

def main():
    global window
    if not glfw.init():
        sys.exit()
    window = glfw.create_window(800, 600, "Flujo Óptico con Matriz de Puntos", None, None)
    if not window:
        glfw.terminate()
        sys.exit()
    glfw.make_context_current(window)
    glViewport(0, 0, 800, 600)
    init_opengl()
    cap = cv2.VideoCapture(0)
    while not glfw.window_should_close(window):
        ret, frame = cap.read()
        if ret:
            handle_optical_flow(frame)
            frame_with_matrix = draw_matrix_on_frame(frame)
            cv2.imshow("Matriz de Puntos", frame_with_matrix)
        draw_scene()
        glfw.poll_events()
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    glfw.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()