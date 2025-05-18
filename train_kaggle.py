# Simple YOLOv8m training for Kaggle
import os
import sys
os.system("pip install -q ultralytics")

import yaml
from ultralytics import YOLO

# Thông số training - thay đổi dễ dàng
EPOCHS = 100  # Số lượng epoch
BATCH_SIZE = 16  # Kích thước batch
IMAGE_SIZE = 640  # Kích thước ảnh đầu vào

# Tạo file YAML tạm thời
dataset_config = {
    'path': './',  # Đường dẫn gốc của dataset, thay đổi nếu cần
    'train': 'train/images',  # Thư mục chứa ảnh training
    'val': 'val/images',      # Thư mục chứa ảnh validation
    'test': 'test/images',    # Thư mục chứa ảnh test
    'nc': 10,                 # Số lượng class
    'names': ['cahukho', 'canhcai', 'canhchua', 'com', 'dauhusotca', 'gachien', 'raumuongxao', 'thitkho', 'thitkhotrung', 'trungchien']
}

# Lưu config vào file tạm thời
with open('temp_data.yaml', 'w') as f:
    yaml.dump(dataset_config, f, default_flow_style=False)

print("Đã tạo file config tạm thời")

# Tải mô hình
model = YOLO('yolov8m.pt')  

# Train mô hình
model.train(
    data='temp_data.yaml', 
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMAGE_SIZE,
    device=0, 
    project='food_detection',
    name='yolov8m_run',
    exist_ok=True,
    pretrained=True,
)

metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")

# Dự đoán trên tập test
model.predict(source=os.path.join(dataset_config['path'], dataset_config['test']), conf=0.25, save=True)

model.export(format='onnx')

print("Training hoàn tất. Mô hình được lưu vào food_detection/yolov8m_run/weights/") 
