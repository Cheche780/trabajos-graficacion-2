import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt
import sys
import math
import cv2
import numpy as np

# Variables globales para el movimiento de la cámara
camera_x = 0
camera_y = 0
window = None

def init():
    """Configuración inicial de OpenGL"""
    glClearColor(0.5, 0.8, 1.0, 1.0)  # Fondo azul cielo
    glEnable(GL_DEPTH_TEST)           # Activar prueba de profundidad
    glMatrixMode(GL_PROJECTION)
    gluPerspective(60, 1.0, 0.1, 100.0)  # Perspectiva
    glMatrixMode(GL_MODELVIEW)

def draw_ground():
    """Dibuja un plano para representar el suelo"""
    glBegin(GL_QUADS)
    glColor3f(0.3, 0.3, 0.3)  # Gris oscuro
    glVertex3f(-20, 0, 20)
    glVertex3f(20, 0, 20)
    glVertex3f(20, 0, -20)
    glVertex3f(-20, 0, -20)
    glEnd()

def draw_cube():
    """Dibuja un cubo (edificio básico)"""
    glBegin(GL_QUADS)
    glColor3f(0.8, 0.5, 0.2)  # Marrón para las caras
    # Frente
    glVertex3f(-1, 0, 1)
    glVertex3f(1, 0, 1)
    glVertex3f(1, 2, 1)
    glVertex3f(-1, 2, 1)
    # Atrás
    glVertex3f(-1, 0, -1)
    glVertex3f(1, 0, -1)
    glVertex3f(1, 2, -1)
    glVertex3f(-1, 2, -1)
    # Lados
    for dx in (-1, 1):
        glVertex3f(dx, 0, -1)
        glVertex3f(dx, 0, 1)
        glVertex3f(dx, 2, 1)
        glVertex3f(dx, 2, -1)
    # Arriba
    glColor3f(0.9, 0.6, 0.3)
    glVertex3f(-1, 2, -1)
    glVertex3f(1, 2, -1)
    glVertex3f(1, 2, 1)
    glVertex3f(-1, 2, 1)
    glEnd()

def draw_scene():
    """Dibuja la escena con edificios y suelo"""
    global camera_x, camera_y

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(20 + camera_x, 15 + camera_y, 25,  # Posición de la cámara
              0, 0, 0,    # Punto al que mira
              0, 1, 0)    # Vector hacia arriba

    draw_ground()  # Dibujar el suelo

    # Dibujar edificios en círculo
    num_buildings = 8
    radius = 10
    for i in range(num_buildings):
        angle = 2 * math.pi * i / num_buildings
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        glPushMatrix()
        glTranslatef(x, 0, z)
        glRotatef(math.degrees(angle), 0, 1, 0)
        draw_cube()
        glPopMatrix()

    glfw.swap_buffers(window)

def process_optical_flow(cap):
    """Procesa el flujo óptico para mover la cámara"""
    global camera_x, camera_y

    ret, frame1 = cap.read()
    if not ret:
        return
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        avg_flow = np.mean(flow, axis=(0, 1))
        camera_x += avg_flow[0] * 0.1
        camera_y -= avg_flow[1] * 0.1
        prvs = next
        cv2.waitKey(10)

def main():
    global window

    if not glfw.init():
        sys.exit()
    
    width, height = 800, 600
    window = glfw.create_window(width, height, "Edificios y Suelo", None, None)
    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)
    glViewport(0, 0, width, height)
    init()

    cap = cv2.VideoCapture(0)

    import threading
    optical_flow_thread = threading.Thread(target=process_optical_flow, args=(cap,))
    optical_flow_thread.start()

    while not glfw.window_should_close(window):
        draw_scene()
        glfw.poll_events()

    glfw.terminate()
    cap.release()
    optical_flow_thread.join()

if __name__ == "__main__":
    main()