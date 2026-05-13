# ASL Hand Gesture Recognition System

A real-time American Sign Language (ASL) hand gesture recognition system built using Python, TensorFlow, OpenCV, and MobileNetV2.

The project can:

Collect hand gesture datasets using a webcam
Train a deep learning model on custom ASL data
Detect hand signs in real-time
Convert detected letters into sentences
Speak the generated sentence using text-to-speech


# Features
Real-Time Hand Detection

Uses webcam input and hand tracking to detect a single hand in real time.

Dataset Collection

Capture and save hand gesture images for training.

Deep Learning Model

Uses MobileNetV2 transfer learning for fast and accurate gesture recognition.

Sentence Builder

Allows users to create words and sentences from predicted letters.

Text-to-Speech

Speaks the generated sentence using the system voice.

Prediction Smoothing

Uses averaging to reduce flickering and unstable predictions.


# Technologies Used

Python
TensorFlow
Keras
OpenCV
CVZone
NumPy


# Project Structure

<img width="848" height="443" alt="image" src="https://github.com/user-attachments/assets/64c01e36-669c-47e7-9683-c7fba4bf46a3" />



# How It Works
Step 1 — Collect Dataset

Run:

python datacol.py
Controls
Press S to save hand images

Images are saved inside:

abc_data/<label_name>/

Example:

abc_data/A/
abc_data/B/
Step 2 — Train Model

Run:

python model.py

The script:

Loads dataset images
Creates training and validation data
Applies data augmentation
Trains MobileNetV2
Saves the best model

Saved files:

asl_best_model.keras
asl_best_model1.h5
Step 3 — Run Real-Time Prediction

Run:

python final.py

The webcam will open and start detecting hand gestures.


# Keyboard Controls
Key	Action
S	Save detected letter
Space	Add space
Backspace	Delete last character
C	Clear sentence
Enter	Speak sentence
Q	Quit program


# Model Details
Base Model

The project uses:

MobileNetV2
ImageNet pretrained weights

Input Size
224 x 224
Techniques Used
Transfer Learning
Data Augmentation
Prediction Smoothing
Class Weight Balancing
Early Stopping
Learning Rate Reduction


# Dataset Tips

For better accuracy:

Use proper lighting
Keep background clean
Capture multiple hand angles
Keep hand fully visible
Collect balanced samples for all classes


# Requirements
Python 3.9+
Webcam
Windows OS (for PowerShell text-to-speech)


# Screenshots
<img width="802" height="566" alt="image" src="https://github.com/user-attachments/assets/8c0377e4-116e-4174-b4c0-7bcd7150b732" />
<img width="555" height="252" alt="image" src="https://github.com/user-attachments/assets/d5f5042d-6e21-41d8-ac9e-ece85f6a7012" />
<img width="450" height="382" alt="image" src="https://github.com/user-attachments/assets/d8e6ab46-0a9f-46b0-99c6-11a6b8520443" />
<img width="602" height="289" alt="image" src="https://github.com/user-attachments/assets/cdeb9046-0e51-4d99-b705-818d364a08f5" />



# Author

## Himanshu Godwal
BTech Student 
at Manipal University Jaipur


This project is for educational purposes.
