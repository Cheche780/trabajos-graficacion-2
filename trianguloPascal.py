import pygame
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
