from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='v3/data.yaml', epochs=10, imgsz=640)

print("Training Complete")
