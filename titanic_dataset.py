# Install if needed
# pip install kagglehub pandas scikit-learn matplotlib seaborn

import kagglehub
from kagglehub import KaggleDatasetAdapter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -----------------------------
# Load Dataset
# -----------------------------

file_path = "Titanic-Dataset.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "yasserh/titanic-dataset",
    file_path,
)

print("First 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# -----------------------------
# Data Preprocessing
# -----------------------------

# Select useful features
X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]

# Fill missing Age values
X['Age'] = X['Age'].fillna(X['Age'].mean())

# Target variable
y = df['Survived']

# -----------------------------
# Split Dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nTraining Accuracy:", round(train_accuracy * 100, 2), "%")
print("Testing Accuracy:", round(test_accuracy * 100, 2), "%")

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_test_pred)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# Display Confusion Matrix
# -----------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Survived", "Survived"]
)

disp.plot()

plt.title("Titanic Dataset - Decision Tree")
plt.show()