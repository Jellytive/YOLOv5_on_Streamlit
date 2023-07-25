import torch
import cv2
import numpy as np
import streamlit as st

# Cargar el modelo pre-entrenado
model = torch.hub.load('ultralytics/yolov5', 'custom', 
                       path='E:\\7mo_semestre\\Proyecto\\YOLOV5s\\yolov5\\y5s_hg_full_ima_ep8.pt')

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

# Función para procesar el frame y realizar la detección de objetos
def process_frame(frame):
    # Realizar la detección
    detect = model(frame)

    # Renderizar la detección en el frame
    render_img = np.squeeze(detect.render())

    return render_img

# Configurar Streamlit
st.title("YOLOv5 Object Detection")

# Crear un espacio vacío para mostrar la imagen de la cámara
frame_placeholder = st.empty()

# Iniciar el bucle de captura y detección de objetos
while True:
    # Realizar la lectura del frame de la captura de video
    ret, frame = cap.read()

    # Verificar si la captura fue exitosa
    if ret:
        # Procesar el frame y realizar la detección de objetos
        result_frame = process_frame(frame)

        # Mostrar el resultado en Streamlit
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
        #st.image(result_frame, channels="BGR", use_column_width=True)

    # Leer el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
