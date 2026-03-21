# Final-Year-Project-Breast-Cancer-Detection-Used VGG16 model and Tkinter

## Overview
This project is a breast cancer detection system developed using a VGG16-based deep learning model and the Tkinter graphical user interface (GUI). The system can classify mammogram images into two categories which are **benign** and **malignant**.

The project applies transfer learning to improve image classification performance and provides a simple desktop application with using Python that allows users to upload a mammogram image and receive a prediction result.

## Objectives
The main objectives of this project are:
- To develop a deep learning model for mammogram image classification.
- To classify breast cancer images into benign and malignant categories.
- To build a simple Tkinter-based GUI for prediction.
- To evaluate the system using standard classification metrics.

## Features
- Mammogram image classification with using VGG16.
- Binary classification: Benign & Malignant
- Transfer learning with fine-tuning
- Tkinter GUI for image upload and prediction
- Image validation before prediction
- Evaluation metrics with using accuracy, precision, recall, F1-score, confusion matrix, ROC curve and AUC

## Model Information
The classification model is used **VGG16** which is a pretrained convolutional neural network.

### Architecture Summary
- Input size: **256 × 256 × 3**
- Base model: **VGG16 (pretrained on ImageNet, include_top=False)**
- Global Average Pooling layer
- Dense layer with 256 units and ReLU activation
- Batch Normalization
- Dropout layer (0.4)
- Dense layer with 64 units and ReLU activation
- Dropout layer (0.2)
- Output layer with 1 unit and sigmoid activation

### Training Strategy
The model training was performed in two phases:
1. **Feature Extraction Phase**  
   The VGG16 convolutional base was frozen and only the custom classifier layers were trained.
2. **Fine-Tuning Phase**  
   Selected upper convolutional layers were unfrozen to adapt the model more effectively to mammogram images.

### Installation
- Please install the needed Python libraries from the requirements.txt

### how to run
- Download the project folder, then run the:
   - app.py file
- The selected model will be included in the same folder. When the load_model successful, there will return the graphical user interface (GUI).

### dataset structure
- CBIS-DDSM dataset.
- Remove Background dataset.
- Combination dataset included CBIS-DDSM original images.


### project structure
- README.md
- Requirements.txt
- .gitignore
- app.py
- VGG16.h5
  
### author
- TEO SHI KAI
