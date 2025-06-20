from ultralytics import YOLO
import os
from collections import Counter
import cv2

# Cargamos el modelo nano
model = YOLO("yolo11n.pt")

# Definimos una lista con las clases que queremos encontrar
vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]

def predict_image(image_path,image_index=""):
    # Definimos nombre de archivo de salida
    filename = os.path.basename(image_path)
    filename = "\\bounding_boxes"+filename.split(".")[0] + "_prediction.jpg"

    # Ejecutamos el modelo sobre la imagen
    results = model(image_path)

    # Obtenemos las clases detectadas (índices -> nombres)
    detections = results[0].boxes.cls.cpu().numpy()
    detected_classes = [model.names[int(i)] for i in detections]

    # Mostrar y guardar imagen con predicciones
    results[0].show()
    results[0].save(filename=filename)

    # Contamos cuántas veces aparece cada clase
    count_vehicle = Counter(detected_classes)

    # Verificamos si hay algún vehículo
    flag = any(clase in detected_classes for clase in vehicle_classes)

    if flag:
        print(f"\n\nVehículos detectados en la imagen:")
        for type in vehicle_classes:
            if type in count_vehicle:
                print(f" - {type}: {count_vehicle[type]}")

        total = sum(count_vehicle[type] for type in vehicle_classes if type in count_vehicle)
        print(f"Total de vehículos: {total}")

    else:
        print("La imagen no muestra vehículos de ningún tipo.")


def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

# Verificar si se abrió correctamente el archivo de video
    if not cap.isOpened():
        print("Error abriendo el archivo de video.")
        exit()

    # Crear una carpeta para guardar los fotogramas (si no existe)
    import os
    output_folder = "fotogramas"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Contador de fotogramas
    frame_count = 0

    # Leer los fotogramas del video
    while True:
        ret, frame = cap.read()  # ret = True si el frame fue leído correctamente, frame = el fotograma
        if ret:  # Si se leyó el frame, guardarlo
            frame_path = os.path.join(output_folder, f"fotograma_{frame_count}.png")
            cv2.imwrite(frame_path, frame)
            predict_image(frame_path,str(frame_count))
            frame_count += 1

        else: 
            break

    # Liberar los recursos
    cap.release()