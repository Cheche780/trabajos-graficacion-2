import cv2
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import sys

# Variables globales
window = None
x_opengl = 0.0
y_opengl = 0.0
scale = 0.2
angle_x = 0
angle_y = 0  

video = cv2.VideoCapture(0)

def init():
    """Inicializa OpenGL"""
    glClearColor(0.0, 0.0, 0.0, 1.0) 
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 640 / 480, 1.0, 50.0) 

    glMatrixMode(GL_MODELVIEW)

def draw_prism(x, y, scale, angle_x, angle_y):
    """Dibuja un prisma rectangular con gestos"""
    glPushMatrix()
    glTranslatef(x, y, -5.0)  
    glScalef(scale, scale * 1.5, scale) 
    glRotatef(angle_x, 1, 0, 0) 
    glRotatef(angle_y, 0, 1, 0)  


    glBegin(GL_QUADS)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1, 1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, 1, 1)
    glVertex3f(1, 1, 1)

    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(1, -1, 1)
    glVertex3f(-1, -1, 1)
    glVertex3f(-1, -1, -1)
    glVertex3f(1, -1, -1)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, -1, 1)
    glVertex3f(1, -1, 1)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1, -1, -1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, -1, 1)

    glColor3f(0.0, 1.0, 1.0)
    glVertex3f(1, 1, -1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, -1, 1)
    glVertex3f(1, -1, -1)
    glEnd()

    glPopMatrix()

def process_frame():
    """Procesa el video y actualiza las coordenadas del prisma"""
    global x_opengl, y_opengl, angle_x, angle_y
    ret, frame = video.read()
    if not ret:
        return

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (15, 15), 0)
    _, thresh = cv2.threshold(frame_blur, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)


        new_x_opengl = (x + w / 2) / frame.shape[1] * 2 - 1
        new_y_opengl = -((y + h / 2) / frame.shape[0] * 2 - 1)


        dx = new_x_opengl - x_opengl
        dy = new_y_opengl - y_opengl


        x_opengl = new_x_opengl
        y_opengl = new_y_opengl
        angle_x = dy * 20  
        angle_y = dx * 20 
    else:

        angle_x, angle_y = 0, 0

def main():
    """Función principal"""
    global window


    if not glfw.init():
        print("Error: GLFW no pudo inicializarse")
        sys.exit()

    width, height = 640, 480
    window = glfw.create_window(width, height, "Prisma controlado con la mano", None, None)

    if not window:
        print("Error: No se pudo crear la ventana GLFW")
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)
    glViewport(0, 0, width, height)
    init()


    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()


        process_frame()
        draw_prism(x_opengl, y_opengl, scale, angle_x, angle_y)

        glfw.swap_buffers(window)
        glfw.poll_events()


    glfw.terminate()
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()