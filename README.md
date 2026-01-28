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


