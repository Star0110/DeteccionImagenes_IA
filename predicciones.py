from ultralytics import YOLO


model = YOLO('runs/detect/train2/weights/best.pt')
preds = model('Programa5/datasets/dataset_vehiculos/images/test')