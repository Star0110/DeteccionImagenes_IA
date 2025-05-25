from ultralytics import YOLO

# 1. Cargar el modelo (nano, el más pequeño)
model = YOLO("yolo11n.pt")
# Predecir una imagen

results = model("imagenes_carros\WhatsApp Image 2025-05-24 at 08.45.01.jpeg")

# Mostrar el resultado (con las cajas dibujadas)
results[0].show()

# Guardar el resultado si quieres
results[0].save(filename="resultado.jpg")