import os, sys
sys.path.append(os.getcwd())
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.train_model import ModelTrainer
import pandas as pd

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        self.model_data_path_config = DataTransformationConfig()

    def start_training(self):
        try:
            logging.info("Starting re - training")

            self.data_ingestion.initiate_data_ingestion()

            self.data_transformation.prepared_data()

            df_train = pd.read_csv(self.model_data_path_config.train_data_path) 
            df_test = pd.read_csv(self.model_data_path_config.test_data_path) 

            self.model_trainer.initiate_model_trainer(df_train, df_test)

            logging.info("Training Completed")

        except Exception as e:
            logging.info("Error Occured")
            raise CustomException(e, sys)