import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.utils import LOGGER  # Import LOGGER

LOGGER.setLevel(50)


model_path = "models/public/anti-spoof-mn3/anti-spoof-mn3.onnx"
session = ort.InferenceSession(model_path)

yolo_model = YOLO("models/yolov8n.pt", verbose=False)  # Using YOLOv8 nano for speed

def preprocess(img):
    """Preprocess image for the anti-spoof model"""
    img = cv2.resize(img, (128, 128))  # Resize to 80x80 (as required by anti-spoof-mn3)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Open webcam
cap = cv2.VideoCapture(1)
default_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
cap.set(cv2.CAP_PROP_BRIGHTNESS, default_brightness-20)
while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    
    phone_detected = False
    results = yolo_model(frame)
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())  # Get class ID
            if cls_id == 67:  # Class ID 67 = "Cell Phone" in COCO dataset
                phone_detected = True
    
    img = preprocess(frame)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})[0]  # Run inference
    
    is_real = output[0][0] < 0.1  # Threshold (adjust if needed)
    print(f"IS REAL : {is_real}  |  CONFIDENCE : {output[0][0]}")
    # Display result
    if is_real == True and phone_detected == False:
        text = "REAL"
    else:
        if phone_detected:
            text = "FAKE (Phone detected)"
        else:
            text = "FAKE"
    color = (0, 255, 0) if text == "REAL" else (0, 0, 255)
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imshow("Face Anti-Spoofing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()