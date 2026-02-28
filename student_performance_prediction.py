# ===============================================
# Student Performance Prediction Using ML
# ===============================================

# 1. Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 2. Create Sample Dataset (Manually created for demo)

data = {
    "Attendance": [90, 60, 75, 30, 85, 50, 95, 40, 70, 65],
    "Study_Hours": [5, 2, 3, 1, 6, 2, 7, 1, 4, 3],
    "Previous_Marks": [80, 45, 60, 30, 85, 50, 90, 35, 65, 55],
    "Assignment_Score": [85, 40, 70, 20, 90, 50, 95, 25, 75, 60],
    "Result": ["Pass", "Fail", "Pass", "Fail", "Pass", "Fail", "Pass", "Fail", "Pass", "Pass"]
}

df = pd.DataFrame(data)

# 3. Convert Result into Numerical Form (Encoding)
df["Result"] = df["Result"].map({"Fail": 0, "Pass": 1})

# 4. Define Features (X) and Target (y)

X = df[["Attendance", "Study_Hours", "Previous_Marks", "Assignment_Score"]]
y = df["Result"]

# 5. Split Dataset into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Create Decision Tree Model

model = DecisionTreeClassifier()

# 7. Train the Model

model.fit(X_train, y_train)

# 8. Make Predictions

y_pred = model.predict(X_test)

# 9. Calculate Accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# 10. Display Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# ===============================================
# 11. Graph 1 - Attendance vs Result
# ===============================================

plt.scatter(df["Attendance"], df["Result"])
plt.xlabel("Attendance")
plt.ylabel("Result (0=Fail, 1=Pass)")
plt.title("Attendance vs Result")
plt.show()

# ===============================================
# 12. Graph 2 - Study Hours vs Result
# ===============================================

plt.scatter(df["Study_Hours"], df["Result"],alpha=0.5)
plt.xlabel("Study Hours")
plt.ylabel("Result (0=Fail, 1=Pass)")
plt.title("Study Hours vs Result")
plt.show()

# ===============================================
# 13. Sample Prediction
# ===============================================

sample_student = [[30, 4, 65, 40]]  # Attendance, Study Hours, Previous Marks, Assignment

prediction = model.predict(sample_student)

if prediction[0] == 1:
    print("Prediction: Student will PASS")
else:
    print("Prediction: Student will FAIL")
    