from flask import Flask, request, render_template
import os
import numpy as np
import pandas as pd
import sklearn
from joblib import load
import warnings

app = Flask(__name__)

# Force sklearn version compatibility
sklearn.__version__ = "1.5.2"
warnings.filterwarnings("ignore", category=UserWarning)

# Load model at startup
try:
    predict_pipeline = load('artifacts/predict_pipeline.joblib')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"! Model loading failed: {str(e)}")
    raise

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', results=None)
    else:
        try:
            form_data = request.form
            
            # Validate required fields
            required_fields = ['gender', 'ethnicity', 'parental_level_of_education', 
                              'lunch', 'test_preparation_course',
                              'reading_score', 'writing_score']
            
            for field in required_fields:
                if not form_data.get(field):
                    return render_template('home.html', 
                                        error_message=f"Please fill in all required fields",
                                        form_data=form_data,
                                        results=None)
            
            # Validate scores
            try:
                reading_score = float(form_data.get('reading_score'))
                writing_score = float(form_data.get('writing_score'))
            except ValueError:
                return render_template('home.html', 
                                    error_message="Scores must be valid numbers",
                                    form_data=form_data,
                                    results=None)
            
            if not (0 <= reading_score <= 100) or not (0 <= writing_score <= 100):
                return render_template('home.html', 
                                    error_message="Scores must be between 0 and 100",
                                    form_data=form_data,
                                    results=None)
            
            # Prepare data for prediction
            data = {
                'gender': [form_data.get('gender')],
                'race_ethnicity': [form_data.get('ethnicity')],
                'parental_level_of_education': [form_data.get('parental_level_of_education')],
                'lunch': [form_data.get('lunch')],
                'test_preparation_course': [form_data.get('test_preparation_course')],
                'reading_score': [reading_score],
                'writing_score': [writing_score]
            }
            
            pred_df = pd.DataFrame(data)
            results = predict_pipeline.predict(pred_df)
            
            formatted_result = round(float(results[0]), 2)
            
            return render_template('home.html', 
                                results=formatted_result,
                                form_data=form_data)
            
        except Exception as e:
            return render_template('home.html', 
                                error_message=f"An error occurred: {str(e)}",
                                form_data=form_data if 'form_data' in locals() else None,
                                results=None)

def run_server():
    if os.name == 'nt':  # Windows
        from waitress import serve
        print("Running Waitress server on Windows...")
        serve(app, host="0.0.0.0", port=8080)
    else:  # Linux/Mac (Render.com)
        print("Running Gunicorn server on Render...")
        app.run(host="0.0.0.0", port=10000)

if __name__ == "__main__":
    run_server()