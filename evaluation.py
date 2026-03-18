from ultralytics import YOLO

model = YOLO("yolo11n.pt")

metrics = model.val(
    data="dataset/data.yaml",
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.7,
    device="mps"
)

print(metrics)