📘 Student Performance Prediction Using Machine Learning
📌 Project Overview
This project predicts whether a student will Pass or Fail using Machine Learning based on academic and performance-related features.
The model is trained using a Decision Tree Classifier from Scikit-learn.

🎯 Objective
To build a classification model that predicts student performance based on:
    -Attendance (%)
    -Study Hours (per day)
    -Previous Marks
    -Assignment Score

🧠 Problem Type
This is a Supervised Machine Learning Classification Problem, where:
    -Input → Student academic features
    -Output → Pass (1) or Fail (0)

🛠 Technologies Used
    -Python
    -Pandas
    -Scikit-learn
    -Matplotlib

📊 Dataset Information
The dataset contains the following columns:
    -Feature	Description
    -Attendance	Student attendance percentage
    -Study_Hours	Daily study hours
    -Previous_Marks	Marks obtained in previous exams
    -Assignment_Score	Assignment performance
    -Result	Pass or Fail (Target Variable)
    
⚙️ Steps Performed
Data Creation using Pandas
Data Preprocessing
Converted categorical output (Pass/Fail) into numerical values (1/0)
Feature & Target Separation
Train-Test Split (70% Training, 30% Testing)
Model Training using Decision Tree Classifier
Model Evaluation using:
Accuracy Score
Confusion Matrix
Data Visualization using Matplotlib
Sample Student Prediction

🤖 Model Used
Decision Tree Classifier
Reason for selection:
    -Easy to understand and interpret
    -Works well for classification problems
    -Suitable for small datasets
    -No need for feature scaling
    
📈 Model Performance
Accuracy Score displayed after testing
Confusion Matrix used to evaluate predictions

📊 Visualizations Included
Attendance vs Result
Study Hours vs Result
These graphs help in understanding the relationship between features and student performance.

🧪 Sample Prediction
Example student input:
    -Attendance = 30
    -Study Hours = 4
    -Previous Marks = 65
    -Assignment Score = 40

The model predicts whether the student will Pass or Fail based on learned patterns.

▶️ How to Run the Project
Install required libraries:
    -pip install pandas scikit-learn matplotlib

Run the Python file:
    -python student_performance_prediction.py

📌 Future Improvements
Use larger real-world dataset
Add more features (Participation, Internal Marks, etc.)
Try advanced models (Random Forest, Logistic Regression)
Deploy as a Web App
