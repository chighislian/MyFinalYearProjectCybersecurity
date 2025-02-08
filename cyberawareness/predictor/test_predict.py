import joblib

# Load the model
MODEL_PATH = "D:/myCyberProject/cyberawareness/predictor/awareness_model.pkl"
model = joblib.load(MODEL_PATH)

# Test inputs
test_data = [
    [5, 0, 10, 7, 10, 10, 0, 0, 0, 10, 7, 3, 3, 3],  # Example input
    [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],  # Expect HIGH
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Expect LOW
]

# Make predictions
for i, data in enumerate(test_data):
    prediction = model.predict([data])[0]
    print(f"Test Case {i+1}: Prediction -> {prediction}")