import os
from TrainData import training_data
from DataPreprocess import TrainDataPreprocess
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
            logging.info(f"There is an error in the Training Model Module __inti__ function.Please check below. /n {str(e)}")

    def separate_train_test(self):
        try:
            logging.info("Entered into separate train test function.")
            self.df = self.preprocessor.remove_unnecessary_feature(self.data)
            self.preprocessor.check_null_duplicate(self.df)
            self.traindata = self.preprocessor.preprocess_rename_feature(self.df)
            logging.info("Exiting from separate train test function.")
            return self.traindata
        except Exception as e:
            logging.info(f"There is an error in the separate_train_test function.Please check below. /n {str(e)}")