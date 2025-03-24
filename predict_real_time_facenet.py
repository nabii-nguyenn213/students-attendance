import cv2
import torch
import numpy as np
import joblib
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(keep_all=True, device=device)

facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

classifier = joblib.load("models/face_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


cap = cv2.VideoCapture(1)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    frame=cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)


            face = rgb_frame[y1:y2, x1:x2]

            if face.shape[0] > 0 and face.shape[1] > 0:
                try:

                    face_tensor = transform(face).unsqueeze(0).to(device)


                    with torch.no_grad():
                        embedding = facenet(face_tensor).cpu().numpy().flatten().reshape(1, -1)


                    pred_label = classifier.predict(embedding)
                    pred_student = label_encoder.inverse_transform(pred_label)[0]


                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    probs = classifier.predict_proba(embedding)
                    confidence = np.max(probs) * 100
                    cv2.putText(frame, f"{pred_student} ({confidence:.2f}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

               
                except Exception as e:
                    print("Error:", e)


    cv2.imshow("Real-Time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng camera
cap.release()
cv2.destroyAllwindows()