import cv2
import os
from datetime import datetime
import time
import csv
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
        self.load_models(model_path=model_path)

    def load_models(self, model_path):
        self.yolo_model = YOLO(model_path+"/yolov8n.pt")
        self.session = ort.InferenceSession(model_path+"/public/anti-spoof-mn3/anti-spoof-mn3.onnx")

    def frame_processing(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        output = self.session.run(None, {input_name: img})[0]  
        
        self.conf = output[0][0]
        is_real = self.conf > self.threshold 
        
        # Display result
        if is_real == True and phone_detected == False:
            return True
        if phone_detected:
            print("Phone detected")
        return False 

    def get_conf(self):
        return self.conf

class Student_Classification:
    def __init__(self, model_path):
        self.load_models(model_path=model_path)

    def load_models(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.classifier = joblib.load(model_path+"/face_classifier.pkl")
        self.label_encoder = joblib.load(model_path+"/label_encoder.pkl")

    def predict(self, frame, boxes):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # boxes, _ = self.mtcnn.detect(rgb_frame)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]
                # face = rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if face.shape[0] > 0 and face.shape[1] > 0:
                    try:
                        face_tensor = transform(face).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            embedding = self.facenet(face_tensor).cpu().numpy().flatten().reshape(1, -1)
                        pred_label = self.classifier.predict(embedding)
                        pred_student = self.label_encoder.inverse_transform(pred_label)[0]


                        self.conf = self.classifier.predict_proba(embedding)
                        self.conf = np.max(self.conf) * 100
                        print(f"PREDICT LABEL : {pred_label} | PREDICT STUDENT : {pred_student}")
                        return pred_label, pred_student
                    except Exception as e:
                        print("Error:", e)

    def get_conf(self):
        return self.conf

class Student_Attendance:

    def __init__(self, threshold):
        self.anti_spoof = Anti_Spoofing(model_path="models", threshold=threshold)
        self.student_classification = Student_Classification(model_path="models")
        self.attended = None
        
    def save_student_image(self, student_id, image, base_folder="Student_Attendance_Image"):
        # Get current date in YYYY-MM-DD format
        current_date = datetime.now().strftime("%Y-%m-%d")

        date_folder = os.path.join(base_folder, current_date)
        student_folder = os.path.join(date_folder, str(student_id))

        os.makedirs(student_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%H-%M-%S")
        image_path = os.path.join(student_folder, f"{timestamp}.jpg")

        cv2.imwrite(image_path, image)

        # print(f"Image saved at: {image_path}")
        
        return image_path
    
    def save_attendance(self, frame, filename="students.csv"):
        columns = ["Student ID", "Date", "Time", "Image_Path", "Confidence"]
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = time.strftime("%H:%M:%S", time.localtime()) 
        file_exists = os.path.exists(filename)
        if file_exists and os.stat(filename).st_size == 0:
            with open(filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(columns) 

        if file_exists:
            try:
                df = pd.read_csv(filename)
                if ((df["Student ID"] == self.student_id) & (df["Date"] == current_date)).any():
                    print(f"Student {self.student_id} đã được ghi nhận hôm nay. Bỏ qua.")
                    self.attended = True
                    return
            except pd.errors.EmptyDataError:
                print("File CSV rỗng, sẽ ghi lại header.")
        
        # Mở file và ghi dữ liệu mới
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(columns)
                
            if self.confidence > 70.:
                student_image_path = self.save_student_image(student_id=self.student_id, image=frame)
                writer.writerow([self.student_id, current_date, current_time, student_image_path, self.confidence])    
                print(f"Đã lưu thành công vào {filename}")
    

    def run(self, show_fps = False):
        number_of_frame_check_spoof = 10 
        cap = cv2.VideoCapture(1)
        default_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        cap.set(cv2.CAP_PROP_FPS, 60)  
        cap.set(cv2.CAP_PROP_BRIGHTNESS, default_brightness+30)
        # cap.set(cv2.CAP_PROP_EXPOSURE, -4) 
        # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = Manual mode (may not work on all webcams)

        current_number_of_frames_check_spoof = 0
        prev_time = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = self.student_classification.mtcnn.detect(rgb_frame)
            
            if boxes is not None:
                is_real = self.anti_spoof.predict(frame=frame)
                print(f"IS REAL : {is_real} | CONFIDENCE : {self.anti_spoof.get_conf()}")
                if is_real:
                    current_number_of_frames_check_spoof += 1
                    print(current_number_of_frames_check_spoof)
                    cv2.putText(frame, "PLEASE KEEP STEADY", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else: 
                    current_number_of_frames_check_spoof = 0
                    cv2.putText(frame, "SPOOF DETECTED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if current_number_of_frames_check_spoof > number_of_frame_check_spoof:
                    self.student_id = self.student_classification.predict(frame=frame, boxes=boxes)[1]
                    self.confidence = self.student_classification.get_conf()

                    print(f"STUDENT ID : {self.student_id}")
                    print(f"CONFIDENCE : {self.confidence}")
                    current_number_of_frames_check_spoof = 0
                    self.save_attendance(frame=frame, filename="students.csv")
                    
            if self.attended:
                cv2.putText(frame, f'Student ID : {self.student_id}', (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'SUCCESSFUL ATTENDANCE', (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        
            if show_fps:
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                cv2.putText(frame, f'FPS: {int(fps)}', (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            
            cv2.imshow("Student Attendance", frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
                
                
if __name__ == "__main__":
    student_attendance = Student_Attendance(threshold=0.4)
    student_attendance.run(show_fps=False)