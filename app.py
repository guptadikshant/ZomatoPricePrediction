from flask import Flask, app, render_template, request,jsonify
#from Prediction import predictions
import os
import pickle
import logging
os.makedirs("Application_Logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("Application_Logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)

loaded_model_path = os.path.join("BestModel","Model.pkl")
loaded_model = pickle.load(open(loaded_model_path,"rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict",methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        try:
            logging.info("Entered into Predict Route")
            
            online_order = request.form["Online Order"]
            book_table = request.form["Book Table"]
            votes = request.form["Votes"]
            location = request.form["Location"]
            type_of_resturant = request.form["Type of Restaurant"]
            type_of_order = request.form["Type of Order"]
            city_name = request.form["City Name"]
            cusinies = request.form["Cuisines"]
            rating = request.form["Rating of Restaurant"]

            # cost = predictions.prediction_func(online_order, book_table, votes, location, 
            #                                      type_of_resturant, type_of_order, city_name, rating, cusinies)

            print(online_order)
            print(book_table)
            print(votes)
            print(location)
            print(type_of_resturant)
            print(type_of_order)
            print(city_name)
            print(cusinies)
            print(rating)
            

            cost = loaded_model.predict([[online_order,book_table, votes,location,type_of_resturant,type_of_order,city_name,
                                        rating,cusinies]])

            print("cost is....",cost)

            logging.info("Exiting from the prediction route.")
            
            return render_template("results.html",cost=round(cost[0]))
            

        except Exception as e:
            logging.info(f"The error in the prediction route is {str(e)}")

    else:
        return render_template("home.html")



if __name__ == "__main__":
    app.run(debug=True)