"""
BAI702 - Machine Learning II Mini Project
Student Marks Prediction Using Linear Regression Algorithm
-------------------------------------------------------------
Predicts a student's final exam marks from:
    - Study Hours
    - Attendance (%)
    - Assignment Score
    - Previous Test Score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------
# 1. Data Collection
# ---------------------------------------------------------------
df = pd.read_csv("student_performance.csv")
print("Dataset shape:", df.shape)
print(df.head())
print(df.describe())

# ---------------------------------------------------------------
# 2. Data Preprocessing
# ---------------------------------------------------------------
print("\nMissing values:\n", df.isnull().sum())
df = df.drop_duplicates()

# ---------------------------------------------------------------
# 3. Feature Selection
# ---------------------------------------------------------------
X = df[["Study_Hours", "Attendance", "Assignment_Score", "Previous_Test_Score"]]
y = df["Final_Exam_Marks"]

# Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("output_correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. Train-Test Split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# 5. Linear Regression Model Training
# ---------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

# ---------------------------------------------------------------
# 6. Prediction
# ---------------------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------------------
# 7. Performance Evaluation
# ---------------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"  MAE  : {mae:.3f}")
print(f"  MSE  : {mse:.3f}")
print(f"  RMSE : {rmse:.3f}")
print(f"  R2   : {r2:.3f}")

# Actual vs Predicted scatter plot
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, color="royalblue", alpha=0.7, edgecolor="k")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, color="red", linestyle="--", label="Ideal Fit")
plt.xlabel("Actual Final Marks")
plt.ylabel("Predicted Final Marks")
plt.title("Actual vs Predicted Final Marks")
plt.legend()
plt.tight_layout()
plt.savefig("output_actual_vs_predicted.png", dpi=150)
plt.close()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 5))
plt.scatter(y_pred, residuals, color="seagreen", alpha=0.7, edgecolor="k")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Final Marks")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("output_residual_plot.png", dpi=150)
plt.close()

# Study Hours vs Final Marks regression line (single-feature view)
plt.figure(figsize=(6, 5))
plt.scatter(df["Study_Hours"], df["Final_Exam_Marks"], color="orange", alpha=0.6, edgecolor="k")
order = np.argsort(df["Study_Hours"].values)
simple_model = LinearRegression().fit(df[["Study_Hours"]], df["Final_Exam_Marks"])
plt.plot(
    df["Study_Hours"].values[order],
    simple_model.predict(df[["Study_Hours"]]).values[order] if hasattr(simple_model.predict(df[["Study_Hours"]]), "values") else simple_model.predict(df[["Study_Hours"]])[order],
    color="red"
)
plt.xlabel("Study Hours")
plt.ylabel("Final Exam Marks")
plt.title("Study Hours vs Final Exam Marks")
plt.tight_layout()
plt.savefig("output_study_hours_trend.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 8. Final Result - Predict marks for a new student
# ---------------------------------------------------------------
new_student = pd.DataFrame({
    "Study_Hours": [7],
    "Attendance": [90],
    "Assignment_Score": [85],
    "Previous_Test_Score": [78]
})
predicted_marks = model.predict(new_student)[0]
print(f"\nExpected Output Example:")
print(f"  Study Hours: 7, Attendance: 90%, Assignment Score: 85, Previous Test Score: 78")
print(f"  Predicted Final Marks: {predicted_marks:.2f}")

# Save metrics + prediction to a text file for the report
with open("model_results.txt", "w") as f:
    f.write("STUDENT MARKS PREDICTION - LINEAR REGRESSION RESULTS\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Dataset size: {df.shape[0]} students, {df.shape[1]} columns\n")
    f.write(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}\n\n")
    f.write("Model Coefficients:\n")
    for feature, coef in zip(X.columns, model.coef_):
        f.write(f"  {feature}: {coef:.4f}\n")
    f.write(f"  Intercept: {model.intercept_:.4f}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"  MAE  : {mae:.3f}\n")
    f.write(f"  MSE  : {mse:.3f}\n")
    f.write(f"  RMSE : {rmse:.3f}\n")
    f.write(f"  R2   : {r2:.3f}\n\n")
    f.write("Sample Prediction:\n")
    f.write("  Input -> Study Hours: 7, Attendance: 90%, Assignment Score: 85, Previous Test Score: 78\n")
    f.write(f"  Predicted Final Marks: {predicted_marks:.2f}\n")

print("\nAll outputs saved successfully.")
