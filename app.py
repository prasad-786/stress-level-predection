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

    X = data.drop("stress_level", axis=1)
    y = data["stress_level"]

    tree_clf = DecisionTreeClassifier(max_depth=7, random_state=100)
    tree_clf.fit(X, y)
    
    # Store class names for UI display
    classes = encoder.classes_
except Exception as e:
    print(f"Error loading dataset: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Extract features in the exact order the model expects
            features = [
                'anxiety_level', 'mental_health_history', 'depression',
                'headache', 'sleep_quality', 'breathing_problem',
                'living_conditions', 'academic_performance', 'study_load',
                'future_career_concerns', 'extracurricular_activities'
            ]
            
            user_data = [float(request.form.get(f)) for f in features]
            prediction = tree_clf.predict([user_data])[0]
            result = encoder.inverse_transform([prediction])[0]

            return render_template('result.html', stress_level=result)
        except (ValueError, TypeError):
            return render_template('error.html', error_message="Please ensure all fields are filled with valid numbers.")

    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)