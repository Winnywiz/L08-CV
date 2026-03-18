from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
yaml_path = "dataset/data.yaml"
# Train the model
if __name__ == '__main__':
    #results = model.train(data=yaml_path, epochs=100, imgsz=640)
    model.tune(data=yaml_path, epochs=5, iterations=10
               , optimizer="AdamW", plots=False, save=True, val=True,device="mps")