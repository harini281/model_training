
# Import Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================
# Load Iris Dataset
# =========================
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Display Dataset Information
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# =========================
# Prepare Features and Target
# =========================
X = iris.data
y = iris.target

# =========================
# Split Dataset
# 80% Training
# 20% Testing
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Train KNN Model
# =========================
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# =========================
# Make Predictions
# =========================
y_pred = model.predict(X_test)

# =========================
# Evaluate Model
# =========================
print("\nTraining Accuracy:")
print(model.score(X_train, y_train))

print("\nTesting Accuracy:")
print(model.score(X_test, y_test))

print("\nOverall Accuracy:")
print(accuracy_score(y_test, y_pred))

# =========================
# Confusion Matrix
# =========================
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# Classification Report
# =========================
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
