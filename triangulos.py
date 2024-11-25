import glfw
from OpenGL.GL import glClear, glClearColor, glBegin, glEnd, glVertex2f, glColor3f, GL_COLOR_BUFFER_BIT, GL_QUADS

def main():
    # Inicializar GLFW
    if not glfw.init():
        return

    # Crear la ventana
    window = glfw.create_window(500, 500, "OpenGL con GLFW", None, None)
    if not window:
        glfw.terminate()
        return

    # Hacer el contexto de OpenGL actual
    glfw.make_context_current(window)

    # Establecer el color de fondo
    glClearColor(0.0, 0.0, 0.0, 1.0)

    # Bucle principal de renderizado
    while not glfw.window_should_close(window):
        # Limpiar la pantalla
        glClear(GL_COLOR_BUFFER_BIT)

        # Dibujar un triángulo
        glBegin(GL_QUADS)
        glColor3f(1.0, 0.0, 0.0)  # Rojo
         # Rojo
        glVertex2f(-0.5,  0.5)  # Vértice superior izquierdo
        glColor3f(0.0, 1.0, 0.0)
        glVertex2f( 0.5,  0.5)  # Vértice superior derecho
        glColor3f(0.0, 0.0, 1.0)  # Rojo
        glVertex2f( 0.5, -0.5)  # Vértice inferior derecho
        glVertex2f(-0.5, -0.5) 

        glEnd()

        # Intercambiar buffers y procesar eventos
        glfw.swap_buffers(window)
        glfw.poll_events()

    # Terminar GLFW
    glfw.terminate()

if __name__ == "__main__":
    main()