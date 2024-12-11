import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt
from OpenGL.GLU import gluNewQuadric, gluCylinder

import sys

def init():
    """Configuración inicial de OpenGL"""
    glClearColor(0.5, 0.8, 1.0, 1.0)  # Fondo azul cielo
    glEnable(GL_DEPTH_TEST)  # Activar prueba de profundidad

    # Configuración de la perspectiva
    glMatrixMode(GL_PROJECTION)
    gluPerspective(60, 1.0, 0.1, 100.0)  # Campo de visión más amplio
    glMatrixMode(GL_MODELVIEW)

def draw_cube():
    """Dibuja un cubo"""
    glBegin(GL_QUADS)
    glColor3f(0.8, 0.5, 0.2)  # Marrón para todas las caras

    # Frente
    glVertex3f(-1, 0, 1)
    glVertex3f(1, 0, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)

    # Atrás
    glVertex3f(-1, 0, -1)
    glVertex3f(1, 0, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(-1, 1, -1)

    # Izquierda
    glVertex3f(-1, 0, -1)
    glVertex3f(-1, 0, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)

    # Derecha
    glVertex3f(1, 0, -1)
    glVertex3f(1, 0, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, 1, -1)

    # Arriba
    glColor3f(0.9, 0.6, 0.3)  # Color diferente para el techo
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)

    # Abajo
    glColor3f(0.6, 0.4, 0.2)  # Suelo más oscuro
    glVertex3f(-1, 0, -1)
    glVertex3f(1, 0, -1)
    glVertex3f(1, 0, 1)
    glVertex3f(-1, 0, 1)
    glEnd()

def draw_cylinder(radius, height, slices):
    """Dibuja un cilindro"""
    glPushMatrix()
    glTranslatef(0, height / 2, 0)
    quad = gluNewQuadric()
    gluCylinder(quad, radius, radius, height, slices, 1)
    glPopMatrix()

def draw_traffic_light():
    """Dibuja un semáforo en el centro de la escena"""
    glPushMatrix()
    glTranslatef(0, 0, 0)  # Posición central

    # Poste del semáforo
    glColor3f(0.1, 0.1, 0.1)  # Negro
    draw_cylinder(0.2, 5, 20)

    # Caja de luces
    glPushMatrix()
    glTranslatef(0, 5, 0)  # Elevar por encima del poste
    glScalef(0.5, 1, 0.5)
    glColor3f(0.2, 0.2, 0.2)  # Gris oscuro
    draw_cube()
    glPopMatrix()

    # Luces del semáforo
    glPushMatrix()
    glTranslatef(0, 5.5, 0.3)  # Frente de la caja
    radii = [0.15, 0.15, 0.15]
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # Rojo, amarillo, verde
    for i, color in enumerate(colors):
        glColor3f(*color)
        glPushMatrix()
        glTranslatef(0, -0.3 * i, 0)
        draw_cylinder(radii[i], 0.1, 20)
        glPopMatrix()
    glPopMatrix()

    glPopMatrix()

def draw_ground():
    """Dibuja un plano para representar el suelo o calle"""
    glBegin(GL_QUADS)
    glColor3f(0.3, 0.3, 0.3)  # Gris oscuro para la calle

    # Coordenadas del plano
    glVertex3f(-20, 0, 20)
    glVertex3f(20, 0, 20)
    glVertex3f(20, 0, -20)
    glVertex3f(-20, 0, -20)
    glEnd()

def draw_house(height_scale=1.0):
    """Dibuja una casa con una altura ajustable."""
    glPushMatrix()
    glScalef(2, height_scale, 2)  # Escalar verticalmente
    draw_cube()  # Base de la casa
    glPopMatrix()

def draw_scene():
    """Dibuja toda la escena con 4 casas y un semáforo."""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Configuración de la cámara
    gluLookAt(20, 15, 25,  # Posición de la cámara
              0, 0, 0,  # Punto al que mira
              0, 1, 0)  # Vector hacia arriba

    # Dibujar el suelo
    draw_ground()

    # Dibujar las casas en diferentes posiciones y alturas
    positions_and_heights = [
        ((-5, 0, -5), 3.0),  # Casa 1, altura normal
        ((5, 0, -5), 5.5),   # Casa 2, más alta
        ((-5, 0, 5), 7.75),  # Casa 3, más baja
        ((5, 0, 5), 10.0)     # Casa 4, la más alta
    ]

    for pos, height in positions_and_heights:
        glPushMatrix()
        glTranslatef(*pos)  # Mover la casa a la posición actual
        draw_house(height_scale=height)  # Dibujar la casa con altura personalizada
        glPopMatrix()

    # Dibujar el semáforo centrado
    draw_traffic_light()

    glfw.swap_buffers(window)

def main():
    global window

    # Inicializar GLFW
    if not glfw.init():
        sys.exit()

    # Crear ventana de GLFW
    width, height = 800, 600
    window = glfw.create_window(width, height, "Escena con casas y semáforo", None, None)
    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)
    glViewport(0, 0, width, height)
    init()

    # Bucle principal
    while not glfw.window_should_close(window):
        draw_scene()
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
