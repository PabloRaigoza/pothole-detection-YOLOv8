from ultralytics import YOLO

model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n.pt')
model3 = YOLO('yolov8n.pt')
model4 = YOLO('yolov8n.pt')
model5 = YOLO('yolov8n.pt')
model6 = YOLO('yolov8n.pt')
model7 = YOLO('yolov8n.pt')
model8 = YOLO('yolov8n.pt')

# Test final learning rates with Stochastic Gradient Descent
results = model1.train(name='SGD_lrf1'   , data='v5/data.yaml', epochs=300, imgsz=640, device=[0], optimizer='SGD', lr0=0.03, lrf=1)
results = model2.train(name='SGD_lrf0.5' , data='v5/data.yaml', epochs=300, imgsz=640, device=[0], optimizer='SGD', lr0=0.03, lrf=0.5)
results = model3.train(name='SGD_lrf0.03', data='v5/data.yaml', epochs=300, imgsz=640, device=[0], optimizer='SGD', lr0=0.03, lrf=0.04)

# Test final learning rates with AdamW
results = model4.train(name='AdamW_lrf1'   , data='v5/data.yaml', epochs=300, imgsz=640, device=[0], optimizer='AdamW', lr0=0.03, lrf=1)
results = model5.train(name='AdamW_lrf0.5' , data='v5/data.yaml', epochs=300, imgsz=640, device=[0], optimizer='AdamW', lr0=0.03, lrf=0.5)
results = model6.train(name='AdamW_lrf0.03', data='v5/data.yaml', epochs=300, imgsz=640, device=[0], optimizer='AdamW', lr0=0.03, lrf=0.04)

# Baseline models for comparison
results = model7.train(name='SGD_lri0.01'  , data='v5/data.yaml', epochs=300, imgsz=640, device=[0], optimizer='SGD'  , lr0=0.01, lrf=1)
results = model8.train(name='AdamW_lri0.01', data='v5/data.yaml', epochs=300, imgsz=640, device=[0], optimizer='AdamW', lr0=0.01, lrf=1)

print("Training Complete")
