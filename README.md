# 🦠 COVID-19 X-Ray Classification (VGG16 + TensorFlow)

<p align="center">
  <img src="https://github.com/user-attachments/assets/9f5ea4e1-15e9-4e5d-bd76-fa4c56b99162" width="650"/>
</p>


An AI-powered medical image classification system that detects
COVID-19, Normal, and Viral Pneumonia cases using Deep Learning.

---

## 🚀 Project Overview

This project builds a Deep Learning model for classifying chest X-ray images into:

- 🦠 Covid-19
- 🏸 Viral Pneumonia
- ✅ Normal

The system uses Transfer Learning with VGG16 to achieve high accuracy in medical image classification.

---

## 📂 Dataset

 link: 
https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset/data

---

## 🧠 System Architecture

### 1️⃣ Data Preparation

- Loads dataset using Pandas DataFrame
- Converts file paths into structured format
- Splits test set into:
    - Validation set
    - Final test set

### 2️⃣ Data Preprocessing

- Image resizing → 224x224
- RGB color mode
- Categorical encoding
- Batch size = 16
- ImageDataGenerator for feeding model

### 3️⃣ Model Architecture (Transfer Learning)

The model uses:

- Pretrained VGG16 (ImageNet weights)
- Global Max Pooling
- Batch Normalization
- Dense Layer (128 neurons)
- Dropout (0.20)
- Final Softmax Classification Layer

---

## 📊 Training Details

- Image size: 224x224
- Batch size: 16
- Epochs: 25 (with Early Stopping)
- Learning Rate: 0.0005
- Validation monitoring
- EarlyStopping is applied to prevent overfitting
  
---

## 📈 Model Evaluation

After training, the model:

- accuracy: 87.88%

---

## Ready for deployment or integration into:

- Streamlit App

---

## 🧪 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Transfer Learning

---

## ▶️ How to Run

### 1️⃣ Install Dependencies
pip install tensorflow numpy pandas matplotlib scikit-learn


### 2️⃣ Update Dataset Path

Modify:

- train_data_path = '.../Covid19-dataset/train'
  
- test_data_path = '.../Covid19-dataset/test'


### 3️⃣ Run the System

python main.py

---

## 👨‍💻 Author

Ibrahem Sayed  
Ai & Computer Vision Engineer
