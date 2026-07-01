import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Student Mental health.csv")

# Remove timestamp
df = df.drop("Timestamp", axis=1)

# Convert all columns to string
for col in df.columns:
    df[col] = df[col].astype(str)

# Target column
y = df["Do you have Depression?"]

# Feature columns
X = df.drop("Do you have Depression?", axis=1)

# Encode features
encoders = {}

for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save everything
joblib.dump(model, "student_wellness_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("Successfully saved!")