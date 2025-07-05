import os, sys
sys.path.append(os.getcwd())
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.train_data_path = os.path.join('artifacts', 'prepared_train.csv')
        self.test_data_path = os.path.join('artifacts', 'prepared_test.csv')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def transform_data(self, df):
        try:
            df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']
            df['test_passed'] = [1 if x > 210 else 0 for x in df['total_score']] # Passed with a average of > 70 in the 3 test
            X = df.drop(['math_score', 'reading_score', 'writing_score', 'total_score', 'test_passed'], axis=1)
            df_encoded = pd.get_dummies(X, columns=X.columns, dtype=int)
            df_encoded['test_passed'] = df['test_passed']
            return df_encoded
        except Exception as e:
            raise CustomException(e, sys)

    def prepared_data(self):
        try:
            train_df = pd.read_csv('artifacts/train.csv')
            test_df = pd.read_csv('artifacts/test.csv')

            logging.info('Transformation of the data is started')

            train_df = self.transform_data(train_df)
            test_df = self.transform_data(test_df)

            train_df.to_csv(self.data_transformation_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.data_transformation_config.test_data_path, index=False, header=True)

            logging.info('Transformation of the data is completed')

            return (self.data_transformation_config.train_data_path, self.data_transformation_config.test_data_path)
   
        except Exception as e:
            logging.info('Problem in the data transformation')
            raise CustomException(e, sys)