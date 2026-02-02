import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, render_template, request

app = Flask(__name__)

# Mapping dictionary for the output
STRESS_MAP = {
    0: "Low Stress",
    1: "Medium Stress",
    2: "High Stress"
}

# Feature list matching the CSV exactly
features = [
    'anxiety_level', 'mental_health_history', 'depression',
    'headache', 'sleep_quality', 'breathing_problem',
    'living_conditions', 'academic_performance', 'study_load',
    'future_career_concerns', 'extracurricular_activities'
]

# Load and prepare data
try:
    data = pd.read_csv("StressLevelDataset.csv")
    X = data[features] 
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
            # Get data from the form
            user_input = []
            for f in features:
                val = request.form.get(f)
                user_input.append(float(val))
            
            # Predict
            prediction_numeric = tree_clf.predict([user_input])[0]
            
            # Convert numeric 0, 1, 2 into "Low", "Medium", "High"
            result_text = STRESS_MAP.get(prediction_numeric, "Unknown")

            return render_template('Prediction.html', stress_level=result_text)
            
        except Exception as e:
            return render_template('Error.html', error_message=f"Input Error: {str(e)}")

    # Load the form
    return render_template('information.html')

if __name__ == '__main__':
    app.run(debug=True)
