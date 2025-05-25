from ultralytics import YOLO
import os

# Cargamos el modelo nano
model = YOLO("yolo11n.pt")

#Definimos una lista con las clases que queremos encontrar
vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]


def predict_image(image_path):
    # definimos nombre de archivo
    filename=os.path.basename(image_path)
    filename.split(".")[0]
    filename+="_prediction.jpg"
    results=model(image_path)

    detections = results[0].boxes.cls.cpu().numpy()
    detected_classes = [model.names[int(i)] for i in detections]

    results[0].show()
    results[0].save(filename=filename)

    flag = any(class_d in detected_classes for class_d in vehicle_classes)

    if flag:
       print(f"Se detectaron las clases: {detected_classes}")
    else:
        print("La imagen no muestra vehiculos de ningun tipo")

   