"""Q5. Naive Bayes algorithm.

Part A: Gaussian Naive Bayes on the Iris dataset.
Part B: Multinomial Naive Bayes for simple text (spam) classification.
Run: python 05_naive_bayes.py
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- Part A: Gaussian NB ----------
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=7, stratify=iris.target)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
pred = gnb.predict(X_test)
print("=== Gaussian Naive Bayes (Iris) ===")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=iris.target_names))

# ---------- Part B: Multinomial NB for spam ----------
messages = [
    "win a free prize now", "limited offer claim your free cash", "congratulations you won lottery",
    "free entry in weekly contest", "urgent claim your reward now", "cheap loans approved instantly",
    "are we meeting for lunch today", "please send the project report", "call me when you are free",
    "see you at the college tomorrow", "assignment submission is on monday", "happy birthday have a great day",
]
labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 1 = spam, 0 = ham

vec = CountVectorizer()
Xt = vec.fit_transform(messages)
mnb = MultinomialNB()
mnb.fit(Xt, labels)

tests = ["claim your free prize", "project report submission tomorrow"]
print("=== Multinomial Naive Bayes (spam filter) ===")
for t, p in zip(tests, mnb.predict(vec.transform(tests))):
    print(f"'{t}' -> {'SPAM' if p else 'HAM'}")
