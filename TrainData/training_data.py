import os
import pandas as pd
import logging
os.makedirs("Application_Logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("Application_Logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)

def get_data():
    
    try:
        print("entered")
        logging.info("Entered the training_data module")
        train_data_path = os.path.join("EDA", "zomato.csv")
        train_df = pd.read_csv(train_data_path)
        logging.info("Data loaded fine. Exiting this module")
        return train_df
    except Exception as e:
        logging.info(f"Couldn't load data due to {str(e)}")



