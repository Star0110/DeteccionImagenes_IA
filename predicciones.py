from ultralytics import YOLO

##No encuentra el path
model = YOLO('runs/detect/train2/weights/best.pt')
preds = model('Programa5/datasets/dataset_vehiculos/images/test')