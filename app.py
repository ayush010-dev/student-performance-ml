import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- 1. Dataset Generation ---
# Using @st.cache_data to prevent regenerating data on every UI update
@st.cache_data
def get_synthetic_data():
    """Generates a synthetic dataset for testing purposes."""
    np.random.seed(42) # Ensure reproducibility
    
    # Generate 200 random student records
    attendance = np.random.randint(40, 100, 200)
    study_hours = np.random.randint(1, 10, 200)
    previous_marks = np.random.randint(30, 100, 200)
    assignment_score = np.random.randint(40, 100, 200)
    
    # Simple logic to determine Pass (1) or Fail (0)
    # Give different weights to the features
    performance_score = (attendance * 0.3) + (study_hours * 3) + (previous_marks * 0.4) + (assignment_score * 0.2)
    status = np.where(performance_score > 65, 1, 0) # 1 = Pass, 0 = Fail
    
    # Store the generated data in a Pandas DataFrame
    dataset = pd.DataFrame({
        'Attendance': attendance,
        'Study Hours': study_hours,
        'Previous Marks': previous_marks,
        'Assignment Score': assignment_score,
        'Status': status
    })
    return dataset

# Load the data
df = get_synthetic_data()

# --- 2. Machine Learning Model Setup ---
# Separate features (X) and target label (y)
X = df[['Attendance', 'Study Hours', 'Previous Marks', 'Assignment Score']]
y = df['Status']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Calculate model accuracy on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# --- 3. Streamlit Web Interface ---
# Setting up page config (Title & modern layout)
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Prediction System")
st.markdown("This application predicts whether a student will **Pass** or **Fail** based on their Attendance, Study Hours, Previous Marks, and Assignment Score. It uses a Machine Learning Decision Tree model.")

# --- Sidebar Elements ---
st.sidebar.header("📝 Enter Student Details")

# Input sliders and number fields for features
user_attendance = st.sidebar.slider("Attendance (%)", min_value=0, max_value=100, value=75, help="Percentage of classes attended")
user_study_hours = st.sidebar.number_input("Study Hours (per day)", min_value=0, max_value=24, value=4, help="Hours spent studying self-paced")
user_prev_marks = st.sidebar.slider("Previous Marks (%)", min_value=0, max_value=100, value=65, help="Marks obtained in the previous semester")
user_assignment = st.sidebar.slider("Assignment Score (%)", min_value=0, max_value=100, value=70, help="Average assignment score")

# Put the inputs into a DataFrame pattern matching the trained model
user_input_df = pd.DataFrame({
    'Attendance': [user_attendance],
    'Study Hours': [user_study_hours],
    'Previous Marks': [user_prev_marks],
    'Assignment Score': [user_assignment]
})

st.subheader("📋 Current Student Inputs")
st.dataframe(user_input_df, hide_index=True)


# --- 4. Prediction Logic ---
# Creating the "Predict Result" Button
if st.button("Predict Result", type="primary", use_container_width=True):
    # Predict using our trained model
    prediction = model.predict(user_input_df)
    
    st.subheader("🎯 Result")
    # Display the result with appealing UI messages
    if prediction[0] == 1:
        st.success("✅ **PASS** - Great job! The student is on track and likely to pass.")
    else:
        st.error("❌ **FAIL** - Require more preparation. Study harder!")

st.divider()

# --- 5. Model Metrics Display ---
st.subheader("📊 Model Evaluation & Metrics")

# Show Accuracy
st.info(f"**Model Accuracy:** {accuracy * 100:.2f}%")

# Create two columns for metrics & visual graphics
col1, col2 = st.columns(2)

with col1:
    # Confusion Matrix
    st.write("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail', 'Pass'])
    disp.plot(cmap='Blues', ax=ax_cm, colorbar=False)
    st.pyplot(fig_cm)

with col2:
    st.markdown("""
    **Understanding the Matrix:**
    - **Top-Left**: True Negatives (Model correctly predicted Fail)
    - **Bottom-Right**: True Positives (Model correctly predicted Pass)
    - **Top-Right**: False Positives
    - **Bottom-Left**: False Negatives
    """)

st.divider()

# --- 6. Matplotlib Graphs ---
st.subheader("📈 Dataset Visualizations")

st.markdown("Below are two visual graphs to understand the generated synthetic data better.")

# Graph 1: Scatter plot (Attendance vs Previous Marks)
st.write("**1. Attendance vs Previous Marks**")
fig1, ax1 = plt.subplots(figsize=(8, 4))
# Plotly scatter showing Pass (1) and Fail (0) in different colors
scatter = ax1.scatter(df['Attendance'], df['Previous Marks'], c=df['Status'], cmap='RdYlGn', alpha=0.8, edgecolors='k')
ax1.set_xlabel("Attendance (%)")
ax1.set_ylabel("Previous Marks (%)")
ax1.set_title("How Attendance & Previous Marks influence outcome")
# Create Legend
legend1 = ax1.legend(*scatter.legend_elements(), title="Status (0: Fail, 1: Pass)", loc="lower right")
ax1.add_artist(legend1)
st.pyplot(fig1)

# Graph 2: Bar Chart (Average Study Hours per Status)
st.write("**2. Average Study Hours (Passed vs Failed)**")
fig2, ax2 = plt.subplots(figsize=(8, 4))
avg_study_time = df.groupby('Status')['Study Hours'].mean()
ax2.bar(['Failed (0)', 'Passed (1)'], avg_study_time.values, color=['#ff6b6b', '#4ecdc4'], edgecolor='k')
ax2.set_xlabel("Student Outcome")
ax2.set_ylabel("Average Study Hours")
ax2.set_title("Average Study Hours for Passing vs Failing Students")
# Add values on top of bars
for i, v in enumerate(avg_study_time.values):
    ax2.text(i, v + 0.1, f"{v:.1f} hrs", ha='center', fontweight='bold')
st.pyplot(fig2)
