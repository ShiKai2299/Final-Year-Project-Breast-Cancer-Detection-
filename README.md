# Final-Year-Project-Breast-Cancer-Detection-Using VGG16 model and Tkinter

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
- Mammogram image classification using VGG16.
- Binary classification: Benign & Malignant
- Transfer learning with fine-tuning
- Tkinter GUI for image upload and prediction
- Image validation before prediction
- Evaluation metrics including accuracy, precision, recall, F1-score, confusion matrix, ROC curve and AUC

## Model Information
The classification model is uses **VGG16** which is a pretrained convolutional neural network.

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

## Installation
- Please install the required Python libraries from the **Requirements.txt** file.
- Install the files needed:
  - app.py
  - VGG16.h5

## How to run
1. Download the project folder
2. Download the model **(VGG16.h5)** from my Google Drive link **(https://drive.google.com/drive/folders/1WFC_A-qQ-O6O-CvKlHXH6U0XVpJ9826s?usp=drive_link)**
3. Paste the model **(VGG16.h5)** file into the project file.
4. Run the **app.py**

! In case the app.py can't open because you did not follow the 1,2 and 3 step. 
   * Condition
      * Selected model should be included in the same folder.
      * In case **load_model** successful, the graphical user interface (GUI) will open.

## Dataset structure
- CBIS-DDSM dataset.
- Remove Background dataset.
- Combination dataset including CBIS-DDSM original images.


## Project structure
- README.md
- Requirements.txt
- .gitignore
- app.py
- VGG16.h5

## Link
- Google Drive Link - **https://drive.google.com/drive/folders/1WFC_A-qQ-O6O-CvKlHXH6U0XVpJ9826s?usp=drive_link**
- YouTube Link - **https://youtu.be/zvrPI8KGGPs**
  
## Author
Teo Shi Kai
