import os
import numpy as np
import pandas as pd
from TrainData import training_data
from DataPreprocess import TrainDataPreprocess
from BestModelFinder import tuner
import BestModelFinder.tuner_optuna as tuner_optuna
from sklearn.model_selection import train_test_split
import logging
os.makedirs("Application_Logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("Application_Logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)

class TrainModel:

    def __init__(self):
        try:
            logging.info("Entered into Training Model Module")
            self.data = training_data.get_data()
            self.preprocessor = TrainDataPreprocess.Preprocessor()
        except Exception as e:
            logging.info(f"There is an error in the Training Model Module __init__ function.Please check below. \n {str(e)}")

    def separate_train_test(self):
        try:
            logging.info("Entered into separate train test function.")
            self.df = self.preprocessor.remove_unnecessary_feature(self.data)
            self.preprocessor.check_null_duplicate(self.df)
            self.traindata = self.preprocessor.preprocess_rename_feature(self.df)
            X = self.traindata.drop("Average Cost",axis=1)
            y = self.traindata["Average Cost"]

            self.x_train,self.x_valid,self.y_train,self.y_valid = train_test_split(X,y,test_size=0.3,random_state=30)

            logging.info("Train Data is separated in training and validation set. Exiting from separate train test function.")
            
        except Exception as e:
            logging.info(f"There is an error in the separate_train_test function.Please check below. \n {str(e)}")

    def preprocess_train_data(self):

        try:
            logging.info("Entered into preprocess_train_data function.")
            self.x_train_enc,self.x_valid_enc = self.preprocessor.encoding_features(self.x_train,self.x_valid)
            self.x_train_imp,self.x_valid_imp,self.y_train_imp,self.y_valid_imp = self.preprocessor.imputing_features(self.x_train_enc,
                                                                                                                    self.x_valid_enc,
                                                                                                                    self.y_train,
                                                                                                                    self.y_valid)
            logging.info("Exiting from preprocess_train_data function.")

            return self.x_train_imp,self.x_valid_imp,self.y_train_imp,self.y_valid_imp

        except Exception as e:
            logging.info(f"There is some error in the preprocess_train_data function. Please check below. \n {str(e)}")

    def select_best_model(self):

        logging.info("Entered into Best model Select function.")

        try:
            tuner_obj = tuner.ModelTuner()

            tuner_obj.get_train_adjr2score(self.x_train_imp,np.ravel(self.y_train_imp))

            tuner_obj.get_valid_adjr2score(self.x_valid_imp,np.ravel(self.y_valid_imp))

            tuner_obj.save_best_model()

            logging.info("Got the score for each model and selected best model out of it. Exiting the function.")

        except Exception as e:
            logging.info(f"There is some error in the select_best_model function. Please check below. \n {str(e)}")

    

