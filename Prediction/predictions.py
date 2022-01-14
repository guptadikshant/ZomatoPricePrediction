import pickle
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
def prediction_func(online_order, book_table, votes, location, type_of_resturant, type_of_order, city_name,rating,cusinies):

    try:

        logging.info("Entered into Prediction making function.")

        loaded_model_path = os.path.join("BestModel","Model.pkl")
        loaded_model = pickle.load(open(loaded_model_path,"rb"))

        labelencoder = pickle.load(open("TrainedEncoders\LabelEncoder.pkl","rb"))
        ordinalencoder = pickle.load(open("TrainedEncoders\OrdinalEncoder.pkl","rb"))

        pred_dict = {
            "online_order" : online_order,
            "book_table" : book_table,
            "votes" : votes,
            "location" : location,
            "Type of Resturant" : type_of_resturant,
            "Type of Order" : type_of_order,
            "City Name" : city_name,
            "rating" : rating,
            "Cusinies_ID" : cusinies
        }

        pred_dict["Cusinies_ID"],pred_dict["Type of Order"],pred_dict["City Name"],pred_dict["location"], pred_dict["online_order"], pred_dict["book_table"], pred_dict["Type of Resturant"] = labelencoder.fit_transform([pred_dict["Cusinies_ID"],pred_dict["Type of Order"],
                                                                                                                                                                                                                pred_dict["City Name"],pred_dict["location"], pred_dict["online_order"], 
                                                                                                                                                                                                                pred_dict["book_table"], pred_dict["Type of Resturant"]])

        pred_df = pd.DataFrame(pred_dict,index=[0])

        # pred_df["Cusinies"] = labelencoder.fit_transform(pred_df["Cusinies"])
        # pred_df["Type of Order"] = labelencoder.fit_transform(pred_df["Type of Order"])
        # pred_df["City Name"] = labelencoder.fit_transform(pred_df["City Name"])
        # pred_df["location"] = labelencoder.fit_transform(pred_df["location"])

        # pred_df["Cusinies", "Type of Order","City Name", "location" ] = labelencoder.fit_transform(pred_df["Cusinies", "Type of Order","City Name", "location" ])

        # pred_df[["Cusinies", "Type of Order","City Name", "location"]] = pred_df[["Cusinies", "Type of Order","City Name", "location"]].apply(labelencoder.fit_transform)

        print("the values are...")
        print(pred_df["Cusinies_ID"])
        print( pred_df["Type of Order"])
        print(pred_df["City Name"])
        print(pred_df["location"])
        print(pred_df["online_order"])
        print(pred_df["book_table"])
        print(pred_df["Type of Resturant"])

        # pred_df[["online_order", "book_table","Type of Resturant"]] = ordinalencoder.fit_transform(pred_df[["online_order", "book_table","Type of Resturant"]])

        print(pred_df)

        avg_cost = loaded_model.predict(pred_df.values)

        logging.info(f"The average cost for 2 people is : {avg_cost}")

        return avg_cost

    except Exception as e:
        logging.info(f"There is some error in the prediction_func function. The erro is : \n {str(e)}")
