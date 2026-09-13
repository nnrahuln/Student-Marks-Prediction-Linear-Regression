import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page title
st.title("Student Marks Prediction")
st.write("Predict Final Exam Marks using Linear Regression")

# Load dataset
df = pd.read_csv("student_performance.csv")

# Features and target
X = df[[
    "Study_Hours",
    "Attendance",
    "Assignment_Score",
    "Previous_Test_Score"
]]

y = df["Final_Exam_Marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# User input
st.header("Enter Student Details")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=90.0
)

assignment_score = st.number_input(
    "Assignment Score",
    min_value=0.0,
    max_value=100.0,
    value=85.0
)

previous_test_score = st.number_input(
    "Previous Test Score",
    min_value=0.0,
    max_value=100.0,
    value=78.0
)

# Prediction button
if st.button("Predict Marks"):

    new_student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Assignment_Score": [assignment_score],
        "Previous_Test_Score": [previous_test_score]
    })

    prediction = model.predict(new_student)

    st.success(
        f"Predicted Final Exam Marks: {prediction[0]:.2f}"
    )