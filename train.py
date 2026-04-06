import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import json

# Ensure the model directory exists
os.makedirs("model", exist_ok=True)

# Load the advanced dataset
data_path = "data/churn_advanced.csv"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

df = pd.read_csv(data_path)

# Prepare Features (X) and Target (y)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Save the model
model_path = "model/model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# Save metrics
metrics = {
    "accuracy": accuracy,
    "feature_names": X.columns.tolist(),
    "feature_importances": model.feature_importances_.tolist(),
    "classification_report": report
}

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Model trained with accuracy: {accuracy:.2f}")
print("Model and metrics saved successfully!")
