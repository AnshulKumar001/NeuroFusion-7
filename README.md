# 🧠 NeuroFusion-7: Neural Network–Based Data Fusion

This project builds a **multi-class medical image classification system** using deep learning. It classifies brain MRI images into **dementia stages and brain tumor types** using **EfficientNetB0 + SMOTE + Neural Network**.

---

## 🚀 Project Overview

The model predicts **7 brain disease categories**:

- MildDemented  
- ModerateDemented  
- VeryMildDemented  
- NonDemented  
- Glioma  
- Meningioma  
- Pituitary  

Key ideas used:
- Transfer Learning with **EfficientNetB0**
- Feature-based classification
- **SMOTE** to fix class imbalance
- Fully connected neural network for final prediction

---

## 🧰 Tech Stack

- Python  
- TensorFlow / Keras  
- EfficientNetB0 (ImageNet pretrained)  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- NumPy, Matplotlib, Pandas

---

## 📁 Dataset Structure

After extracting `final.zip`, the dataset directory should look like this:

final/
├── MildDemented/
├── ModerateDemented/
├── VeryMildDemented/
├── NonDemented/
├── glioma/
├── meningioma/
└── pituitary/


Each folder contains brain MRI images in `.jpg / .jpeg / .png` format.

---

## ⚙️ Methodology

### 1️⃣ Image Loading
- Images resized to **150 × 150**
- Converted to NumPy arrays
- Labels extracted from folder names

### 2️⃣ Feature Extraction
- Used **EfficientNetB0** (without top layer)
- Global Average Pooling applied
- Converts images into feature vectors

### 3️⃣ Label Encoding
- Class labels encoded using `LabelEncoder`

### 4️⃣ Class Imbalance Handling
- **SMOTE** applied on training features
- Balances all 7 classes

### 5️⃣ Model Architecture

Input Layer
↓
Dense (256, ReLU)
↓
Dropout (0.3)
↓
Dense (128, ReLU)
↓
Dropout (0.3)
↓
Dense (Softmax – 7 classes)


### 6️⃣ Training Configuration
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Epochs: 20  
- Batch Size: 32  

---

## 📊 Results

| Metric | Accuracy |
|------|----------|
| Training Accuracy | **93.02%** |
| Testing Accuracy | **92.75%** |

✔ High accuracy  
✔ Low overfitting  
✔ Balanced predictions due to SMOTE  

---

## 💾 Model Saving

The trained model is saved as:
model.save("model.keras")

##▶️ How to Run

Upload final.zip to Google Colab

Open the notebook

Run all cells sequentially

Model trains and saves automatical


