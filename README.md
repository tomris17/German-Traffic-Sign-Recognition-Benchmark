#  Traffic Sign Recognition System (GTSRB)

This repository contains an end-to-end deep learning project that classifies traffic signs using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. It includes a custom-built Convolutional Neural Network (CNN) trained with TensorFlow/Keras and an interactive web interface powered by Streamlit.

---

##  Project Overview

* **Dataset:** GTSRB (43 different traffic sign categories)
* **Frameworks:** TensorFlow / Keras, OpenCV, Pandas, NumPy, Scikit-Learn
* **Web UI:** Streamlit
* **Performance:** Over 99% accuracy on validation and test datasets

---

##  Model Architecture

The CNN model is designed with regularized blocks to prevent overfitting:
* **Input Layer:** `(32, 32, 3)` image dimensions
* **Conv Blocks:** `Conv2D` + `BatchNormalization` + `MaxPooling2D` + `Dropout`
* **Dense Layers:** Fully connected layers with `ReLU`, `BatchNormalization`, and `Dropout(0.5)`
* **Output Layer:** `Dense(43, activation='softmax')`
* **Loss Function:** `categorical_crossentropy` | **Optimizer:** `adam`

---

##  Project Structure

```text
├── German_Traffic_Sign_Recognition_Benchmark.ipynb  # Training & EDA Notebook
├── app.py                                          # Streamlit web application
├── traffic_sign_model.keras                         # Saved trained model
├── requirements.txt                                # Required dependencies
└── README.md                                       # Project documentation
