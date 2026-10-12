"""Q4. SVM and Decision Tree classification.

Dataset: Iris (built into scikit-learn, no download needed).
Run: python 04_svm_decision_tree.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# ---------------- SVM ----------------
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_train_s, y_train)
svm_pred = svm.predict(X_test_s)
print("=== SVM (RBF kernel) ===")
print("Accuracy:", round(accuracy_score(y_test, svm_pred), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred, target_names=iris.target_names))

# ---------------- Decision Tree ----------------
dt = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=1)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("=== Decision Tree (max_depth=3) ===")
print("Accuracy:", round(accuracy_score(y_test, dt_pred), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred, target_names=iris.target_names))

plt.figure(figsize=(12, 7))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.savefig("decision_tree.png", dpi=120)
print("Plot saved: decision_tree.png")

# Predict a new flower
sample = [[5.1, 3.5, 1.4, 0.2]]
print("SVM prediction  :", iris.target_names[svm.predict(sc.transform(sample))[0]])
print("Tree prediction :", iris.target_names[dt.predict(sample)[0]])
