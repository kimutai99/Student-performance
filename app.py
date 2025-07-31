from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('home.html') 

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', results=None)
    else:
        try:
            # Get and validate form data
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
            data = CustomData(
                gender=form_data.get('gender'),
                race_ethnicity=form_data.get('ethnicity'),
                parental_level_of_education=form_data.get('parental_level_of_education'),
                lunch=form_data.get('lunch'),
                test_preparation_course=form_data.get('test_preparation_course'),
                reading_score=reading_score,
                writing_score=writing_score
            )
            
            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            # Format the result to 2 decimal places
            formatted_result = round(float(results[0]), 2) if results is not None else None
            
            return render_template('home.html', 
                                results=formatted_result,
                                form_data=form_data)
            
        except Exception as e:
            return render_template('home.html', 
                                error_message=f"An error occurred: {str(e)}",
                                form_data=form_data if 'form_data' in locals() else None,
                                results=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)