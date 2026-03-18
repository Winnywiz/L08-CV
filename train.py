from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # start small (faster)

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)