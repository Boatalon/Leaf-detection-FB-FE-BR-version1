import ultralytics
from ultralytics import YOLO
import os
import cv2

model = YOLO('Model Path : on runs/detect/train/weight/best.pt')
image_path = 'Your image Path'
output_folder = 'C:/Users/Arnon/leaf-object-detection/output'
os.makedirs(output_folder, exist_ok=True)

CLASS_COLORS = {
            0: (255, 0, 0),
            1: (0, 255, 0),
            2: (0, 0, 255)
        }
Class_Fullname = {
            0: "ใบกรางเครือ",
            1: "ใบโพธิ์อินเดีย",
            2: "ใบยางอินเดีย"
        }

image_name = os.path.basename(image_path)
results = model(image_path, verbose=False)
result = results[0]
confidences = result.boxes.conf
classes = result.boxes.cls.tolist()
class_counts = {int(cls): classes.count(cls) for cls in set(classes)}
conForBox  = 0.7 #ปรับค่า confidences ตรงนี้ได้ จะวาดและแสดงจำนวนตามที่เลือก

print("---------------------------------------------------------------------")
if len(class_counts) > 0:
    print(f"ในภาพ {image_name}:")
    for cls_id, count in class_counts.items():
        for conf in confidences:
            if conf < conForBox:
                count = count -1
        class_name = model.names[cls_id]
        fullname = Class_Fullname.get(int(cls_id))
        print(f"ที่ค่า Confidences >= {conForBox*100} :")
        print(f"[{fullname}{class_name}]: {count} ชิ้น")
        print("---------------------------------------------------------------------")
else:
    print(f"ในภาพ {image_name}: ไม่พบวัตถุที่ตรวจจับได้")
    print("---------------------------------------------------------------------")

image = cv2.imread(image_path)


for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, confidences):
    if conf >= conForBox:
        x1, y1, x2, y2 = map(int, box) 
        class_id = int(cls)
        class_name = model.names[class_id]
        color = CLASS_COLORS.get(class_id)
        label = f"{class_name} {conf:.2f}"

        line_thickness = max(1, int(min(image.shape[0], image.shape[1]) / 200))
        font_scale = max(0.5, min(image.shape[0], image.shape[1]) / 1000)
        font_thickness = max(1, int(line_thickness / 2))

        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        cv2.putText(
            image,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )


output_path = os.path.join(output_folder, os.path.basename(image_path))
cv2.imwrite(output_path, image)

print("\nDone! ภาพพร้อม Bounding Box ถูกบันทึกในโฟลเดอร์:", output_folder)
