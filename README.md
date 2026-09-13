# 🎓 Student Marks Prediction Using Linear Regression

## 📌 Project Overview

This project predicts a student's **Final Exam Marks** using the **Linear Regression** machine learning algorithm.

The prediction is based on:

* Study Hours
* Attendance
* Assignment Score
* Previous Test Score

The project also provides a simple **Streamlit web application** where users can enter student details and get predicted marks.

---

## 🎯 Objective

The main objective of this project is to build a machine learning model that can estimate a student's final exam marks based on academic performance and study-related factors.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

---

## 🤖 Machine Learning Algorithm

### Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict a continuous numerical value.

In this project, Linear Regression is used to predict:

**Final Exam Marks**

from:

**Study Hours + Attendance + Assignment Score + Previous Test Score**

---

## 📂 Project Structure

```text
Student_Marks_Prediction_Project/
│
├── app.py
├── student_marks_prediction.py
├── student_performance.csv
├── model_results.txt
│
├── output_actual_vs_predicted.png
├── output_correlation_heatmap.png
├── output_residual_plot.png
├── output_study_hours_trend.png
│
├── Project_Synopsis.docx
├── Student_Marks_Prediction_Report.docx
└── README.md
```

---

## 📊 Dataset Features

| Feature             | Description                   |
| ------------------- | ----------------------------- |
| Study_Hours         | Number of hours studied       |
| Attendance          | Student attendance percentage |
| Assignment_Score    | Assignment marks              |
| Previous_Test_Score | Previous test marks           |
| Final_Exam_Marks    | Final examination marks       |

---

## ⚙️ How to Run the Project

### 1. Install required libraries

Open the VS Code terminal and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### 2. Run the Machine Learning program

```bash
python student_marks_prediction.py
```

### 3. Run the Streamlit Web Application

```bash
streamlit run app.py
```

The application will open in the browser.

Usually the local address is:

```text
http://localhost:8501
```

---

## 🌐 Streamlit Web Application

The web application allows the user to enter:

* Study Hours
* Attendance
* Assignment Score
* Previous Test Score

After clicking **Predict Marks**, the trained Linear Regression model predicts the student's final exam marks.

---

## 📈 Model Evaluation

The project evaluates the model using:

* MAE — Mean Absolute Error
* MSE — Mean Squared Error
* RMSE — Root Mean Squared Error
* R² Score — Coefficient of Determination

The project also generates:

1. Correlation Heatmap
2. Actual vs Predicted Plot
3. Residual Plot
4. Study Hours vs Final Marks Trend

---

## 🔮 Sample Prediction

Example input:

```text
Study Hours: 7
Attendance: 90%
Assignment Score: 85
Previous Test Score: 78
```

The trained model uses these values to predict the student's final exam marks.

---

## 👨‍🎓 Academic Project

**Course:** BAI702 – Machine Learning II
**Project:** Student Marks Prediction Using Linear Regression
**Program:** Artificial Intelligence and Machine Learning
**Semester:** 7th Semester

---

## 📌 Future Enhancements

* Add more student records to the dataset
* Add additional machine learning algorithms
* Improve the Streamlit user interface
* Add student performance charts
* Deploy the Streamlit application online
* Compare Linear Regression with other regression algorithms

---

## 📜 Conclusion

This project demonstrates how **Machine Learning and Linear Regression** can be used to predict student academic performance.

The Streamlit application provides an easy-to-use interface for entering student information and obtaining predicted final exam marks.
