
from django.http import HttpResponse
from django.shortcuts import render, redirect
import joblib
import numpy as np
import os
from django.views.decorators.csrf import csrf_exempt

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'predictor/awareness_model.pkl')
model = joblib.load(MODEL_PATH)

# Mapping for categorical variables
mapping_dict = {
    'q1': {'Very familiar': 5,  'Somewhat familiar': 3, 'Not familiar': 0},
    'q2': { 'I have never heard of it': 0,
            "I have heard of it but don't know what it is": 5,
            'I know what it is and I can identify phishing attempts': 15,
            'I know what it is but find it hard to identify phishing': 10},

    'q3': {'Yes, I know in detail': 10, 'I have a basic understanding': 7,
            "I have heard of it but don't know how it works": 3, "No, I don't know": 0},
    'q4': {'Monthly': 10, 'Every few months': 7, 'Once a year': 3, 'Never': 0},
    'q5': {'Yes, for most accounts': 0, 'Yes, but only a few accounts': 5,
           'No, I use different passwords for all accounts': 10},

    'q6': {'Yes, for all accounts': 10, 'Yes, but for some accounts': 5, 'No': 0},

    'q7': {'Open the attachment to see what it is': 0, 'Delete the email immediately': 5,
           'Scan the attachment with antivirus software before opening': 7,
           'Report it as spam/phishing': 10},
    'q8': {'Yes': 1, 'No': 0},
    'q9': {'Changed passwords': 1, 'Reported it to the relevant authorities': 3,
           'Did nothing': 0, 'Contacted the service provider': 2},
    'q10': {'Click the link and update your information': 0, 'Ignore the email': 5,
            'Mark the email as spam': 7, 'Contact the bank directly to verify the email': 10},
    'q11': {'Run your antivirus software to check for issues': 10,
            'Close the pop-up and continue browsing': 7, 'Restart your computer': 5,
            'Download and install the software immediately': 0},
    'q12': {'Extremely important': 5, 'Very important': 3, 'Somewhat important': 1,
            'Not important': 0},
    'q13': {'Very confident': 5, 'Somewhat confident': 3, 'Not very confident': 1,
            'Not confident at all': 0},
    'q14': {'Yes': 5, 'Maybe': 3, 'No': 0}
}

def home(request):
    return render(request, 'predictor/home.html')


@csrf_exempt
def predict(request):
    if request.method == "POST":
        print("✅ Form submitted!")

        try:
            input_data = []

            # Collect input data from form
            for key in mapping_dict.keys():
                value = request.POST.get(key)
                if value not in mapping_dict[key]:
                    return render(request, 'predictor/predict.html', {'error': f'Invalid input for {key}'})
                input_data.append(mapping_dict[key][value])

            print("🔍 Input Data:", input_data)

            # Convert input data to NumPy array
            input_array = np.array(input_data).reshape(1, -1)

            # Make prediction
            print("🔄 Making prediction...")
            prediction = model.predict(input_array)[0]
            print("🔍 Raw Prediction from Model:", prediction)

            # Check if prediction is numeric or string
            try:
                prediction = float(prediction)  # Convert model output to float
                is_numeric = True
            except ValueError:
                is_numeric = False

            if is_numeric:
                # Model returned a numeric prediction
                score = round(prediction, 2)  # Use the model's numeric output as the score

                # Assign awareness level based on score
                if prediction >= 70:
                    awareness_level = "High Awareness"
                elif 40 <= prediction < 70:
                    awareness_level = "Medium Awareness"
                else:
                    awareness_level = "Low Awareness"

            else:
                # Model returned a category label like "High Awareness"
                awareness_level = prediction  # Directly use model's output
                score = (sum(input_data) / (10 * len(input_data))) * 100  # Normalize score for consistency

            print(f"✅ Storing in session: Level={awareness_level}, Score={score}%")

            # Store in session
            request.session["awareness_level"] = awareness_level
            request.session["score"] = round(score, 2) if score is not None else "N/A"

            print("✅ Redirecting to result page...")
            return redirect("result")

        except Exception as e:
            print(f"❌ Error: {e}")
            return HttpResponse(f"❌ Error: {e}")

    return render(request, "predictor/predict.html")




from django.shortcuts import render, redirect

def result(request):
    # Retrieve session data
    awareness_level = request.session.get("awareness_level")
    score = request.session.get("score")

    print(f"🔍 Retrieved from session -> Awareness Level: {awareness_level}")
    print(f"🔍 Retrieved from session -> Score: {score}")

    # If no session data, redirect back to predict
    if awareness_level is None:
        print("❌ No session data found! Redirecting back to /predict/")
        return redirect("/predict/")

    return render(request, "predictor/result.html", {"awareness_level": awareness_level, "score": score})