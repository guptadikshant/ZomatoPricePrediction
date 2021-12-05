import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer
import logging
os.makedirs("Application_Logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("Application_Logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)

class Preprocessor:

    def __init__(self):
        pass

    def remove_unnecessary_feature(self,data):
        try:
            logging.info("Entered into Train Data Preprocessor Module.")
            logging.info("Entered into remove_unnecessary_feature function. ")
            data.drop(["url","address","name","phone","menu_item","reviews_list","dish_liked"],axis=1,inplace=True)
            logging.info("Unnecessary features removed. Exiting from the function.")
            return data
        except Exception as e:
            logging.info(f"There is an error in the remove_unnecessary_feature function.Please check below. /n {str(e)}")

    def check_null_duplicate(self,data):
        try:
            logging.info("Entered into check_null_duplicate function.")
            duplicate_df = data[data.duplicated(keep=False)]
            duplicate_df.to_csv("Null_Duplicate_Data/TrainData_Duplicate.CSV",index=False)

            feature_with_null = [feature for feature in data.columns if data[feature].isnull().sum() > 0]

            if len(feature_with_null) > 0:

                dataframe_with_null = data[feature_with_null].isnull().sum().to_frame().reset_index()
                dataframe_with_null.columns = ["Feature Name", "Number of Missing Values"]

                dataframe_with_null.to_csv("Null_Duplicate_Data/TrainData_NullValues.CSV", index=False)
                logging.info("Feature Have Some Missing and Duplicate Values.Check Missing Values folder for features having missing values.Exiting the function")

            else:
                logging.info("No Duplicate and Missing Values in any feature.Exiting the function")
        except Exception as e:
            logging.info(f"There is an error in the check_null_duplicate function.Please check below. /n {str(e)}")

    
    def preprocess_rename_feature(self,data):
        try:
            logging.info("Entered into preprocess_rename_feature function.")
            data.rename(columns={"approx_cost(for two people)":"Average Cost",
                                "listed_in(type)" : "Type of Order",
                                "rest_type" : "Type of Resturant",
                                "listed_in(city)" : "City Name"},inplace=True)

            data["rating"] = data["rate"].str.split("/").str.get(0)

            data.drop("rate",axis=1,inplace=True)

            data["Average Cost"] = data["Average Cost"].str.replace(",","")
            data["Average Cost"] = data["Average Cost"].astype("float64")

            new_rate = (data["rating"].shape[0] / data[data["rating"] == "NEW"].shape[0])/5
            new_rate = round(new_rate,1)
            data["rating"]=np.where(data["rating"] == "NEW",new_rate,data["rating"])

            logging.info("All the features are renamed and some preprocessing is complete. Exiing from the function.")

            return data
        except Exception as e:
            logging.info(f"There is an error in the preprocess_rename_feature function.Please check below. /n {str(e)}")

    


        
