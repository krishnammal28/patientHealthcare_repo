"""Q2. Simple Linear Regression (one input variable -> one output variable).

Aim: predict Salary from Years of Experience.
Run: python 02_simple_linear_regression.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset
data = pd.DataFrame({
    "YearsExperience": [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7,
                        3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0,
                        6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5],
    "Salary": [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189,
               63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940,
               91738, 98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872],
})

X = data[["YearsExperience"]]
y = data["Salary"]

# 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print("Intercept (b0):", round(model.intercept_, 2))
print("Slope (b1)    :", round(model.coef_[0], 2))
print("MSE           :", round(mean_squared_error(y_test, y_pred), 2))
print("RMSE          :", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
print("R2 score      :", round(r2_score(y_test, y_pred), 4))

# 5. Predict for a new value
new_exp = 6.5
print(f"Predicted salary for {new_exp} years: {model.predict(pd.DataFrame({'YearsExperience': [new_exp]}))[0]:.2f}")

# 6. Plot
plt.scatter(X_train, y_train, color="blue", label="Train")
plt.scatter(X_test, y_test, color="green", label="Test")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")
plt.legend()
plt.savefig("simple_linear_regression.png", dpi=120)
print("Plot saved: simple_linear_regression.png")
