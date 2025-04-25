# 😄 Face Emotion Detection using Deep Learning

This project is a real-time facial emotion detection system using Convolutional Neural Networks (CNN) trained on the **FER-2013** dataset. It detects human emotions like **Happy**, **Sad**, **Angry**, **Surprised**, etc., directly from webcam input.

---

## 📂 Dataset

- **FER-2013** dataset from Kaggle
- Contains 48x48 grayscale images of faces
- 7 Emotion classes:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral

---

## 🧠 Model Architecture

- Input: 48x48 grayscale image
- CNN Layers: 3 convolutional blocks with BatchNorm & MaxPooling
- Dense Layers: Fully Connected + Dropout
- Output: Softmax layer (7 classes)
- Loss: Categorical Crossentropy
- Optimizer: Adam

---

## 🚀 Features

- Real-time emotion detection using webcam
- Trained on FER-13 dataset
- Supports 7 facial expressions
- Python, TensorFlow & OpenCV based
- Lightweight and fast

---

## 🛠️ Tech Stack

- Python 3.12
- TensorFlow 2.17.1
- Keras
- OpenCV
- NumPy
- Matplotlib

---

## 📦 Installation

```bash
git clone https://github.com/your-username/face-emotion-detection.git
cd face-emotion-detection
pip install -r requirements.txt
"# Face_Emotion_Detection" 
