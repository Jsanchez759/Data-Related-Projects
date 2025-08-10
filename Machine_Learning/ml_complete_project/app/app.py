import sys, os, json
from flask import Flask, request, render_template
sys.path.append(os.getcwd())
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import TrainingPipeline
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
        )
        pred_df = data.prepare_data_to_predict()
        predict_pipeline = PredictPipeline()
        results, confidence_score = predict_pipeline.predict(pred_df)
        result = 'Pass' if results[0] == 1 else 'No Pass'
        return render_template('home.html', results=result, confidence_score=confidence_score[0]*100)
    
    
if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')