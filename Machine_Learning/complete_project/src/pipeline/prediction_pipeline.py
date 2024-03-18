import os, sys, json, pickle
sys.path.append(os.getcwd())
from src.exception import CustomException
from src.logger import logging
from pycaret.classification import predict_model, load_model
import pandas as pd

with open('artifacts/columns.json', 'r') as file:
    columns_names = json.load(file)

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model = load_model('artifacts/model')
            df = predict_model(model, data= features)
            return df['prediction_label'], df['prediction_score']
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course

    def prepare_data_to_predict(self):
        try:
            columns = columns_names['columns'][:-1]
            features = [self.gender, self.race_ethnicity, self.parental_level_of_education, self.lunch, 
                        self.test_preparation_course]

            df = pd.DataFrame(columns=columns)
            df.loc[0] = [0] * len(columns)

            for i in list(df.columns):
                if i in features:
                    df.at[0, i] = 1
                else:
                    df.at[0, i] = 0
            
            return df

        except Exception as e:
            raise CustomException(e, sys)