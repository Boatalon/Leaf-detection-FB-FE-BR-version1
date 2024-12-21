
import ultralytics
from ultralytics import YOLO
from roboflow import Roboflow

# rf = Roboflow(api_key="E0RpFQyXauH2gJ4JG8CX")
# project = rf.workspace("konesann-work-space").project("leaftype-detection")
# version = project.version(5)
# dataset = version.download("yolov8")
                

model = YOLO('yolov8n.pt')
model.info()
model.train(data='C:/Users/Arnon/leaf-object-detection/Leaftype-detection-v3/data.yaml', epochs=50, imgsz=350)

