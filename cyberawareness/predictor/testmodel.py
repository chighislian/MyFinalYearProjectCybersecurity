import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "awareness_model.pkl")

if os.path.exists(MODEL_PATH):
    print("✅ Model file found at:", MODEL_PATH)
else:
    print("❌ Model file NOT found! Check the file path.")



import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
MODEL_PATH = os.path.join(BASE_DIR, "awareness_model.pkl")  # Construct full path

print(f"🔍 Checking model path: {MODEL_PATH}")

# Check if file exists before loading
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    print("❌ Model file not found at", MODEL_PATH)

