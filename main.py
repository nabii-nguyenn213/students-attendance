import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import joblib
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
import pandas as pd

LOGGER.setLevel(50)

class Anti_Spoofing: 
    def __init__(self, model_path, threshold=0.5):
        self.threshold = threshold
        self.get_models(model_path=model_path)

    def get_models(self, model_path):
        self.yolo_model = YOLO(model_path+"/yolov8n.pt")
        self.session = ort.InferenceSession(model_path+"/public/anti-spoof-mn3/anti-spoof-mn3.onnx")

    def frame_processing(self, frame):
        '''
        resize the frame to fit the anti-spoof model.
        '''
        frame = cv2.resize(frame, (128, 128))
        frame = np.transpose(frame, (2, 0, 1)).astype(np.float32)/255.0
        frame = np.expand_dims(frame, axis = 0)
        return frame

    def predict(self, frame):
        phone_detected = False
        results = self.yolo_model(frame)
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())  # Get class ID
                if cls_id == 67:  # Class ID 67 = "Cell Phone" in COCO dataset
                    phone_detected = True
        
        img = self.frame_processing(frame)
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: img})[0]  # Run inference
        
        self.conf = output[0][0]
        is_real = output[0][0] > self.threshold  # Threshold (adjust if needed)
        
        # Display result
        if is_real == True and phone_detected == False:
            return True
        return False 

    def get_conf(self):
        '''
        return : 
            current_frame confidence
        '''
        return self.conf

class Student_Classification:
    def __init__(self, model_path):
        self.get_models(model_path=model_path)

    def get_models(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.classifier = joblib.load(model_path+"/face_classifier.pkl")
        self.label_encoder = joblib.load(model_path+"/label_encoder.pkl")

    def predict(self, frame):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(rgb_frame)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = rgb_frame[y1:y2, x1:x2]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    try:
                        face_tensor = transform(face).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            embedding = self.facenet(face_tensor).cpu().numpy().flatten().reshape(1, -1)
                        pred_label = self.classifier.predict(embedding)
                        pred_student = self.label_encoder.inverse_transform(pred_label)[0]


                        self.conf = self.classifier.predict_proba(embedding)
                        self.conf = np.max(self.conf) * 100

                        return pred_label, pred_student
                    except Exception as e:
                        print("Error:", e)

    def get_conf(self):
        return self.conf

class Student_Attendance:

    def __init__(self):
        self.anti_spoof = Anti_Spoofing(model_path="models", threshold=0.45)
        self.student_classification = Student_Classification(model_path="models")
        
    def save_attendance(self):
        pass

    def run(self):
        number_of_frame_check_spoof = 10
        # number_of_frame_check_spoof variable is used to verify 120 frames all REAL then continued to classifier
        
        cap = cv2.VideoCapture(1)

        current_number_of_frames_check_spoof = 0
        
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)
            ''' Anti-Spoofing section '''
            is_real = self.anti_spoof.predict(frame=frame)
            
            # print(f"IS REAL : {is_real}")
            
            if is_real: 
                current_number_of_frames_check_spoof += 1
                print(current_number_of_frames_check_spoof)
                cv2.putText(frame, "PLEASE KEEP STEADY", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else: 
                current_number_of_frames_check_spoof = 0
                cv2.putText(frame, "SPOOF DETECTED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if current_number_of_frames_check_spoof > number_of_frame_check_spoof:
                student_pred = self.student_classification.predict(frame=frame)[1]
                self.confidence = self.student_classification.get_conf()

                print(f"STUDENT ID : {student_pred}")
                print(f"CONFIDENCE : {self.confidence}")
                current_number_of_frames_check_spoof = 0
                
            cv2.imshow("Student Attendance", frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
                
                
if __name__ == "__main__":
    student_attendance = Student_Attendance()
    student_attendance.run()
