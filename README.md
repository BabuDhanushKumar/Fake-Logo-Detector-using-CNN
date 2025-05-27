# 🧠 Fake-Logo-Detector-using-CNN

**🔍 Overview**

Fake Logo Detector is a Convolutional Neural Network (CNN)-based image classification system designed to distinguish between authentic and counterfeit brand logos. In an age where counterfeit products are rampant online, this project offers a machine learning-based solution to safeguard brand integrity and enhance consumer trust.

This project was developed as part of the Mini Project-II (21AIM56) at the Department of Artificial Intelligence & Machine Learning, New Horizon College of Engineering, Bangalore.

**🎯 Objectives**

Detect fake logos using image classification.

Enhance accuracy using CNNs trained on a balanced dataset of real and fake logos.

Provide a simple user interface for logo uploads and real-time detection.

Deploy the model to offer reliable outputs: "Genuine (1)" or "Potentially Counterfeit (0)".

**🧱 Architecture**

The system follows this high-level pipeline:

Dataset Curation: Collect and label genuine vs. fake logos.

Preprocessing: Resize, normalize, and augment the dataset.

Modeling:

CNN architecture with Conv2D, MaxPooling, Dense, and Dropout layers.

Trained using binary cross-entropy loss and Adam optimizer.

Interface: A Python GUI (interface.py) allows users to upload logos and get real-time results.

**📂 Project Structure**

├── interface.py         # UI for logo upload and prediction
├── logods.py            # Model training and saving
├── test.py              # Load saved model and evaluate
├── /dataset             # Organized dataset (train/valid/genuine/fake)
├── model/               # Saved trained CNN model
└── Report.pdf           # Detailed academic project report

**🧪 Technologies Used**

Python 🐍

TensorFlow/Keras

NumPy

OpenCV

scikit-learn

Tkinter (for GUI)

**📊 Results**

The model achieves high accuracy in differentiating fake and real logos. Output is binary:

1 → Genuine Logo

0 → Potentially Fake Logo

**🔒 Security & Privacy**

The model prioritizes:

Real-time efficiency

Secure data handling

Compliance with data protection best practices

**🚀 Future Enhancements**

Apply transfer learning for more generalization.

Use ensemble learning for improved robustness.

Optimize for edge deployment (e.g., mobile devices).

Integrate explainable AI to make predictions transparent.

Add continuous learning for evolving counterfeit patterns.

**👥 Contributors**

S Vasanth Kumar (1NH21AI116)

M Babu Dhanush Kumar (1NH21AI144)

B Abhilash Reddy (1NH21AI146)

Guide: Mr. S. Gunasekar, Sr. Assistant Professor, AIML Dept., NHCE
