
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Generate a sample dataset
np.random.seed(42)
number_of_students = 500

study_hours = np.random.uniform(1, 10, number_of_students)
attendance = np.random.uniform(50, 100, number_of_students)
previous_score = np.random.uniform(30, 100, number_of_students)
practice_tests = np.random.randint(1, 11, number_of_students)

# Generate scores with some random variation
noise = np.random.normal(0, 5, number_of_students)


final_score = (
    3.0 * study_hours
    + 0.20 * attendance
    + 0.35 * previous_score
    + 1.5 * practice_tests
    + noise
)

final_score = np.clip(final_score, 0, 100)


final_score = np.clip(final_score, 0, 100)

# 2. Create a DataFrame
df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "practice_tests": practice_tests,
    "final_score": final_score
})

# Save dataset for inspection
df.to_csv("student_data.csv", index=False)

# 3. Separate features and target
X = df[[
    "study_hours",
    "attendance",
    "previous_score",
    "practice_tests"
]]
y = df["final_score"]

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("STUDENT PERFORMANCE PREDICTION")
print("--------------------------------")
print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("R2 Score:", round(r2, 2))

# 7. Save the trained model
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nDataset saved as student_data.csv")
print("Model saved as student_model.pkl")
