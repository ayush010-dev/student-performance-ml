🎓 Student Performance Prediction System
~A Machine Learning web application built with Python, Streamlit, and Scikit-Learn that predicts whether a student will Pass or Fail based on academic and behavioral parameters such as attendance, study hours, previous marks, and assignment scores.
The application also provides model evaluation metrics and visualizations to help understand how the model works.

🚀 Features
✅ Predicts student performance (Pass / Fail) using Machine Learning
✅ Uses Decision Tree Classifier for prediction
✅ Interactive Streamlit web interface
✅ User input through sliders and number fields
✅ Displays model accuracy and confusion matrix
✅ Includes dataset visualizations
✅ Uses synthetic dataset generation for testing

🧠 Machine Learning Model
The application uses a Decision Tree Classifier from Scikit-Learn.
Input Features
  -Attendance (%)
  -Study Hours (per day)
  -Previous Marks (%)
  -Assignment Score (%)

Output
  -Pass (1)
  -Fail (0)

The model is trained using 80% training data and 20% testing data.

📊 Model Evaluation
The app shows model performance using:
  -Accuracy Score
  -Confusion Matrix

Confusion matrix helps understand:
Prediction	    Meaning
True Positive	  Correctly predicted Pass
True Negative	  Correctly predicted Fail
False Positive	Predicted Pass but actually Fail
False Negative	Predicted Fail but actually Pass

📈 Visualizations
The application includes two dataset visualizations:
1️⃣ Attendance vs Previous Marks
    -Scatter plot showing how attendance and previous marks affect student performance.

2️⃣ Average Study Hours Comparison
    -Bar chart showing average study hours of students who passed vs failed.

⚙️ Technologies Used
    -Python
    -Streamlit
    -Pandas
    -NumPy
    -Scikit-Learn
    -Matplotlib

📦 Installation
Clone the repository:
  git clone https://github.com/ayush010-dev/student-performance-predictor.git

Navigate to the project folder:
  cd student-performance-predictor

Install required libraries:
  pip install -r requirements.txt

▶️ Run the Application
    -Start the Streamlit app:
       streamlit run app.py
    -The application will open in your browser at:
       http://localhost:8501

🧪 Dataset
The project uses a synthetic dataset generated using NumPy.
It contains 200 student records with the following attributes:
  Feature	Description
  Attendance	Class attendance percentage
  Study Hours	Daily study hours
  Previous Marks	Marks from previous semester
  Assignment Score	Average assignment score
  Status	Pass (1) or Fail (0)

The dataset is generated dynamically each time the app runs.

🎯 Use Cases
    -Academic performance analysis
    -Educational data science projects
    -Machine learning demonstrations
    -College mini projects

📌 Future Improvements
Possible enhancements:
  -Upload real student dataset
  -Add more ML algorithms (Random Forest, SVM)
  -Improve UI with Plotly dashboards
  -Add model explainability (feature importance)
  -Deploy the app on Streamlit Cloud

👨‍💻 Author
Developed as a Machine Learning and Streamlit project for educational purposes.
