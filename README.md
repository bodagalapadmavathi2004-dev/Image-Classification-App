# 🖼️ AI Image Classification App

An AI-powered image classification web application built using Python, TensorFlow, Keras, and Streamlit.

The application uses the pre-trained MobileNetV2 deep learning model with ImageNet weights to identify objects in uploaded or camera-captured images.

---

## 🚀 Features

- 📁 Upload JPG, JPEG, and PNG images
- 📷 Capture images using a webcam
- 🤖 AI-based image classification
- 🧠 Pre-trained MobileNetV2 model
- 📚 ImageNet dataset
- 🔢 1000 image classes
- 🏆 Top 5 predictions
- 📊 Prediction confidence scores
- 📈 Interactive confidence chart
- 📥 Download predictions as CSV
- 🕘 Prediction history
- 🗑️ Clear prediction history
- 📄 Generate downloadable PDF reports
- 🎨 Professional Streamlit interface
- ⚡ Real-time image prediction

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
- Pillow
- ReportLab

---

## 🧠 AI Model

This project uses the pre-trained MobileNetV2 convolutional neural network.

MobileNetV2 is designed for efficient image classification and computer vision applications.

The model uses ImageNet weights and can classify images into 1000 ImageNet categories.

### Model Details

| Property | Value |
|---|---|
| Model | MobileNetV2 |
| Dataset | ImageNet |
| Classes | 1000 |
| Input Size | 224 × 224 |
| Framework | TensorFlow / Keras |

---

## 🔄 How the Application Works

```text
             User
               |
               v
     Upload Image / Camera
               |
               v
        Image Preprocessing
               |
               v
        Resize to 224x224
               |
               v
          MobileNetV2
               |
               v
       ImageNet Prediction
               |
               v
       Top 5 Predictions
               |
        +------+------+
        |      |      |
        v      v      v
     Chart    CSV    PDF
               |
               v
       Prediction History