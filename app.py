import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request

app = Flask(__name__)

# Load and prepare data once when the server starts
try:
    data = pd.read_csv("StressLevelDataset.csv")
    encoder = LabelEncoder()
    data["stress_level"] = encoder.fit_transform(data["stress_level"])

    # List of features in the exact order the model was trained on
    features = [
        'anxiety_level', 'mental_health_history', 'depression',
        'headache', 'sleep_quality', 'breathing_problem',
        'living_conditions', 'academic_performance', 'study_load',
        'future_career_concerns', 'extracurricular_activities'
    ]

    X = data[features] # Ensure X only contains these features in this order
    y = data["stress_level"]

    tree_clf = DecisionTreeClassifier(max_depth=7, random_state=100)
    tree_clf.fit(X, y)
    
    print("Model trained successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get data from the form using the updated feature list
            user_data = [float(request.form.get(f)) for f in features]
            
            # Predict
            prediction = tree_clf.predict([user_data])[0]
            
            # Convert numeric prediction back to original label (e.g., Low/Medium/High)
            result = encoder.inverse_transform([prediction])[0]

            # UPDATED: Using Information.html instead of result.html
            return render_template('Information.html', stress_level=result)
            
        except (ValueError, TypeError) as e:
            return render_template('error.html', error_message="Please ensure all fields are filled with valid numbers.")

    # UPDATED: Using Prediction.html instead of login.html
    return render_template('Prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
