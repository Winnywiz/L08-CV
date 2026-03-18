from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    device="mps",
    lr0= 0.00086,
    momentum= 0.91523,
    weight_decay= 0.0008
)