from flask import Flask, request, render_template
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load model
model = load('artifacts/predict_pipeline.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'gender': [request.form['gender']],
            'race_ethnicity': [request.form['ethnicity']],
            'parental_level_of_education': [request.form['parental_level_of_education']],
            'lunch': [request.form['lunch']],
            'test_preparation_course': [request.form['test_preparation_course']],
            'reading_score': [float(request.form['reading_score'])],
            'writing_score': [float(request.form['writing_score'])]
        }
        
        # Make prediction
        df = pd.DataFrame(data)
        prediction = model.predict(df)[0]
        return render_template('home.html', 
                            prediction=round(prediction, 2),
                            show_result=True)
    
    except Exception as e:
        return render_template('home.html', 
                            error=str(e),
                            show_result=False)

if __name__ == '__main__':
    app.run()