pip install ultralytics
pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="oJZS4HsZTLkmeVBnJUgF")
project = rf.workspace("paathukalam-boys-lf6oa").project("college-bus")
version = project.version(1)
dataset = version.download("yolov8-obb")

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.train(
    data="/content/College-Bus-1/data.yaml",  
    epochs=100,               
    imgsz=640,               
    batch=16,                  
    name="yolov8_custom"     
)

metrics = model.val()  
print(metrics.box.map)  

results = model("/content/WhatsApp Image 2025-02-02 at 23.29.08_22266f73.jpg")  
if results:
    results[0].show()  

pip install pytesseract

from ultralytics import YOLO
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

model = YOLO("/content/runs/detect/yolov8_custom/weights/best.pt")  
cap = cv2.VideoCapture(0)  
MIN_PLATE_WIDTH = 100  
MIN_PLATE_HEIGHT = 30  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  
        confidences = result.boxes.conf.cpu().numpy()  
        for box, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            plate_width = x2 - x1
            plate_height = y2 - y1

            if plate_width >= MIN_PLATE_WIDTH and plate_height >= MIN_PLATE_HEIGHT:
                number_plate_region = frame[int(y1):int(y2), int(x1):int(x2)]

                gray = cv2.cvtColor(number_plate_region, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                text = pytesseract.image_to_string(binary, config='--psm 8')
                cleaned_text = "".join([char for char in text if char.isalnum()])  

                if cleaned_text: 
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Text: {cleaned_text}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real-Time Number Plate Detection", frame)

cap.release()

pip install paddleocr
pip install paddlepaddle

from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR
from google.colab.patches import cv2_imshow

paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')  

model = YOLO("/content/runs/detect/yolov8_custom/weights/best.pt")  
image_path = "/content/bus2.jpg" 
img = cv2.imread(image_path)
results = model(img)

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  
    confidences = result.boxes.conf.cpu().numpy()  

    if len(boxes) == 0:
        print("No number plates detected.")
        continue

    for box, confidence in zip(boxes, confidences):
        x1, y1, x2, y2 = box
        number_plate_img = img[int(y1):int(y2), int(x1):int(x2)]
        ocr_result = paddle_ocr.ocr(number_plate_img, cls=True)
        extracted_text = ""
        if ocr_result:
            for line in ocr_result[0]:
                text = line[1][0] 
                extracted_text += text + " "  
        cleaned_text = "".join([char for char in extracted_text if char.isalnum() or char.isspace()]).strip()
        print("Extracted Text:", cleaned_text)
        cv2_imshow(number_plate_img)
  results = model("/content/bus2.jpg")  

if results:
    results[0].show() 
results = model("/content/bus test.png") 

if results:
    results[0].show()  

results = model("/content/WhatsApp Image 2025-02-02 at 23.29.08_22266f73.jpg")  
if results:
    results[0].show()  
