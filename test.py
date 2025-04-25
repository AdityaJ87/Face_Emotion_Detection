import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"  # Disable Intel extension auto-loading
import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from datetime import datetime
import logging
from torchsummary import summary

# Configuration
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
img_size = 48
device = torch.device('cpu')

# Setup logging
logging.basicConfig(filename='testing.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the model (must match train.py)
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 3 * 3, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
try:
    if not os.path.exists('emotion_model.pth'):
        raise FileNotFoundError("emotion_model.pth not found. Run train.py first.")
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load('emotion_model.pth', map_location=device, weights_only=True))
    model.eval()
    summary(model, (1, img_size, img_size))  # Log model summary
    print("Model loaded successfully")
    logging.info("Model loaded successfully")
except FileNotFoundError as e:
    raise FileNotFoundError(str(e))
except Exception as e:
    raise Exception(f"Model loading failed: {e}")

# Face detection with Haar Cascade
try:
    haar_cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(haar_cascade_path):
        raise FileNotFoundError("haarcascade_frontalface_default.xml not found. Download from: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    if face_cascade.empty():
        raise ValueError("Failed to load Haar Cascade from haarcascade_frontalface_default.xml")
    print("Haar Cascade loaded successfully")
    logging.info("Haar Cascade loaded successfully")
except FileNotFoundError as e:
    raise FileNotFoundError(str(e))
except Exception as e:
    raise Exception(f"Haar Cascade loading failed: {e}")

# Webcam setup
cap = None
for i in range(3):  # Try indices 0, 1, 2
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Webcam opened on index {i}")
        logging.info(f"Webcam opened on index {i}")
        break
if not cap or not cap.isOpened():
    raise ValueError("No webcam accessible")

# Preprocessing
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Real-time detection loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            logging.error("Failed to capture frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (img_size, img_size))
            face_tensor = preprocess(face).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(face_tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred].item() * 100
                label = f"{emotion_labels[pred]} ({confidence:.1f}%)"
                
                # Print and log confidence scores
                print("\nConfidence scores:")
                logging.info("\nConfidence scores:")
                for i, emotion in enumerate(emotion_labels):
                    score = probs[0, i].item() * 100
                    print(f"{emotion}: {score:.1f}%")
                    logging.info(f"{emotion}: {score:.1f}%")
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 100, 100), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)
        
        cv2.imshow('Emotion Detection', frame)
        
        # Save snapshot on 's' key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}_{label.replace(' ', '_')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved snapshot: {filename}")
            logging.info(f"Saved snapshot: {filename}")
        
except Exception as e:
    print(f"Error: {e}")
    logging.error(f"Error: {e}")
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()