import os, sys, json, warnings
sys.path.append(os.getcwd())
from src.exception import CustomException
from src.logger import logging
from pycaret.classification import *
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

warnings.filterwarnings('ignore')

def save_json(dictionary, path):
     try:
        with open(path, 'w') as file:
            json.dump(dictionary, file)
        return None
     
     except Exception as e:
        raise CustomException(e, sys)



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model')
    results_model_file_path = os.path.join('artifacts','results_model.json')
    columns_file_path = os.path.join('artifacts','columns.json')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, df_prepared_test, model):
        try:
            y_true = df_prepared_test['test_passed']
            y_pred = predict_model(model, data = df_prepared_test)['prediction_label']

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            evaluation_metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            }

            logging.info('Completed evaluating model')

            return evaluation_metrics

        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_model_trainer(self, df_prepared_train, df_prepared_test):
        try:
            
            logging.info('Initializing PyCaret setup')
            clf = setup(data = df_prepared_train, target = 'test_passed')
            best_model = compare_models()

            tuned_model = tune_model(best_model, optimize = 'Accuracy', n_iter=50)

            save_model(tuned_model, self.model_trainer_config.trained_model_file_path)

            evaluation_metrics = self.evaluate_model(df_prepared_test, tuned_model)

            save_json(evaluation_metrics, self.model_trainer_config.results_model_file_path)
            save_json({'columns' : list(df_prepared_test.columns)}, self.model_trainer_config.columns_file_path)

            logging.info('Saving and completed the model')

            return (self.model_trainer_config.trained_model_file_path, self.model_trainer_config.results_model_file_path)

        except Exception as e:
            logging.info('Error training model')
            raise CustomException(e, sys)