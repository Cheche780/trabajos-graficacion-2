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