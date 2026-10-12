"""Q3. Logistic Regression.

Aim: classify whether a person will buy a product (0/1) from Age and Estimated Salary.
Run: python 03_logistic_regression.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Synthetic dataset (reproducible)
rng = np.random.default_rng(42)
n = 300
age = rng.integers(18, 60, n)
salary = rng.integers(15000, 150000, n)
score = 0.15 * (age - 38) + 0.00005 * (salary - 80000) + rng.normal(0, 1, n)
purchased = (score > 0).astype(int)
data = pd.DataFrame({"Age": age, "EstimatedSalary": salary, "Purchased": purchased})
print(data.head())

X = data[["Age", "EstimatedSalary"]]
y = data["Purchased"]

# 2. Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

# 3. Train
clf = LogisticRegression()
clf.fit(X_train_s, y_train)

# 4. Evaluate
y_pred = clf.predict(X_test_s)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# 5. Predict for a new customer: age 35, salary 90000
new = sc.transform(pd.DataFrame({"Age": [35], "EstimatedSalary": [90000]}))
print("Prediction for (35, 90000):", clf.predict(new)[0], "| probability:", clf.predict_proba(new)[0].round(3))

# 6. Sigmoid curve illustration
z = np.linspace(-8, 8, 200)
plt.plot(z, 1 / (1 + np.exp(-z)))
plt.axhline(0.5, color="grey", linestyle="--")
plt.title("Sigmoid function used by Logistic Regression")
plt.xlabel("z")
plt.ylabel("P(y=1)")
plt.savefig("logistic_sigmoid.png", dpi=120)
print("Plot saved: logistic_sigmoid.png")
