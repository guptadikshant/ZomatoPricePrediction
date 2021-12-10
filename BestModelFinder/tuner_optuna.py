import os
import pickle
import numpy as np
from TrainingModels import training_models
from sklearn.linear_model import LinearRegression,Lasso,LassoCV,Ridge,RidgeCV,ElasticNet,ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import optuna
import logging
os.makedirs("Application_Logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("Application_Logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)

def model_tuner(x_train,x_valid,y_train,y_valid):

    def objective(trial):

        # train_models_obj = training_models.TrainModel()
        # x_train,x_valid,y_train,y_valid = train_models_obj.preprocess_train_data()

        regressor_name = trial.suggest_categorical('classifier', ['SVR', 'RandomForest'])

        if regressor_name == 'SVR':
            svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
            regressor_obj = SVR(C=svr_c)
        else:
            rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
            regressor_obj = RandomForestRegressor(max_depth=rf_max_depth)

        regressor_obj.fit(x_train, y_train)
        y_pred = regressor_obj.predict(x_valid)

        error = mean_squared_error(y_valid, y_pred)

        return error  

    study = optuna.create_study()  
    study.optimize(objective, n_trials=10)  
    trial = study.best_trial 

    return trial.params

