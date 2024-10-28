import cv2
import numpy as np

# Cargar las máscaras que deseas agregar (asegúrate de que sean PNG con transparencia)
mascaras = [
    cv2.imread('m.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('bigote.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('pug.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('b.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('cre.png', cv2.IMREAD_UNCHANGED)
]

# Verificar si todas las imágenes tienen un canal alfa
for i, mascara in enumerate(mascaras):
    if mascara is None or mascara.shape[2] != 4:
        print(f"Error: La imagen {i+1} no tiene canal alfa o no se pudo cargar.")
        exit()

# Cargar el clasificador preentrenado de rostros
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Capturar video desde la cámara
video = cv2.VideoCapture(0)

# Definir desplazamientos personalizados para cada máscara (x, y)
desplazamientos = [
    (0, 100),   # Desplazamiento para la máscara 1
    (20, 40),  # Desplazamiento para la máscara 2
    (30, 30),  # Desplazamiento para la máscara 3
    (10, 50),  # Desplazamiento para la máscara 4
    (0, 70)    # Desplazamiento para la máscara 5
]

# Definir escalas personalizadas para cada máscara (relativo al tamaño del rostro)
escalas = [
    1.0,  # Escala de la máscara 1
    1.2,  # Escala de la máscara 2
    0.8,  # Escala de la máscara 3
    1.1,  # Escala de la máscara 4
    1.0   # Escala de la máscara 5
]

while True:
    # Leer cada frame del video
    ret, frame = video.read()

    if not ret:
        break

    # Convertir el frame a escala de grises
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar los rostros en el frame
    rostros = face_cascade.detectMultiScale(frame_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Procesar cada rostro detectado
    for i, (x, y, w, h) in enumerate(rostros):
        # Seleccionar una máscara de la lista en función del índice del rostro detectado
        mascara = mascaras[i % len(mascaras)]
        escala = escalas[i % len(escalas)]
        desplazamiento_x, desplazamiento_y = desplazamientos[i % len(desplazamientos)]

        # Redimensionar la máscara según la escala definida
        nuevo_ancho = int(w * escala)
        nuevo_alto = int(h * escala)
        mascara_redimensionada = cv2.resize(mascara, (nuevo_ancho, nuevo_alto))

        # Separar los canales de la máscara: color y alfa (transparencia)
        mascara_rgb = mascara_redimensionada[:, :, :3]
        mascara_alpha = mascara_redimensionada[:, :, 3]

        # Asegurarse de que la máscara alfa sea de tipo uint8
        mascara_alpha = cv2.convertScaleAbs(mascara_alpha)

        # Aplicar el desplazamiento a las coordenadas x e y
        x_nuevo = x + desplazamiento_x
        y_nuevo = y + desplazamiento_y

        # Evitar que la máscara salga del borde de la imagen
        if x_nuevo < 0: x_nuevo = 0
        if y_nuevo < 0: y_nuevo = 0
        if x_nuevo + nuevo_ancho > frame.shape[1]: x_nuevo = frame.shape[1] - nuevo_ancho
        if y_nuevo + nuevo_alto > frame.shape[0]: y_nuevo = frame.shape[0] - nuevo_alto

        # Crear una región de interés (ROI) en el frame donde colocaremos la máscara
        roi = frame[y_nuevo:y_nuevo+nuevo_alto, x_nuevo:x_nuevo+nuevo_ancho]

        # Asegurarse de que la ROI y la máscara tengan el mismo tamaño
        if roi.shape[:2] == mascara_alpha.shape[:2]:
            # Invertir la máscara alfa para obtener la parte del rostro donde se aplicará la máscara
            mascara_alpha_inv = cv2.bitwise_not(mascara_alpha)

            # Enmascarar la región del rostro en la imagen original
            fondo = cv2.bitwise_and(roi, roi, mask=mascara_alpha_inv)

            # Enmascarar la máscara RGB
            mascara_fg = cv2.bitwise_and(mascara_rgb, mascara_rgb, mask=mascara_alpha)

            # Combinar el fondo (parte del rostro sin máscara) y la parte con la máscara
            resultado = cv2.add(fondo, mascara_fg)

            # Reemplazar la región del rostro con la imagen combinada
            frame[y_nuevo:y_nuevo+nuevo_alto, x_nuevo:x_nuevo+nuevo_ancho] = resultado

        else:
            print("Error: El tamaño de la ROI no coincide con la máscara.")

    # Mostrar el frame con las máscaras aplicadas
    cv2.imshow('Video con mascaras', frame)

    # Presionar 'q' para salir del loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
video.release()
cv2.destroyAllWindows()
