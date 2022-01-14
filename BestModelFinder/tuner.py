import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from RegressionMetric import regression_metrics
import logging
os.makedirs("Application_Logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("Application_Logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)

class ModelTuner:

    def __init__(self):

        logging.info("Entered into Model Tuner Class.")

        try:

            logging.info("Loading the respective models.")

            self.lrmodel = LinearRegression()
            self.dtmodel = DecisionTreeRegressor()
            self.rfmodel = RandomForestRegressor()
            self.etmodel = ExtraTreesRegressor()
            self.admodel = AdaBoostRegressor()
            self.gbmodel = GradientBoostingRegressor()
            self.xgmodel = XGBRegressor()
            self.knnmodel = KNeighborsRegressor()
            self.svmmodel = SVR()

            logging.info("All the models loaded fine. Exiting from this function")

        except Exception as e:

            logging.info(f"There is some error in loading the model. The error is : {str(e)}")


    def get_best_param_linear_model(self,x_train,y_train):

        try:
            logging.info("Entering into get best param function for Linear Regression")

            param_linear = {
                "fit_intercept" : ["True","False"],
                "positive" : ["True","False"]
            }

            linear_search = RandomizedSearchCV(self.lrmodel,
                                               param_linear,
                                               cv=5,
                                               n_jobs=-1)

            linear_search.fit(x_train,y_train)

            best_params = linear_search.best_params_
            
            self.lrmodel = LinearRegression(**best_params)

            self.lrmodel.fit(x_train, y_train)

            logging.info(f"The best parameter for Linear Regression are : {best_params}. Exiting the function")

            return self.lrmodel

        except Exception as e:
            logging.info(f"There is some error in the get_best_param_linear_model function. The error is : {str(e)} ")

    
    def get_best_param_decision_tree(self,x_train,y_train):

        try:
            logging.info("Entering into get best param function for Decision Tree")

            param_decision = {
            "criterion": ["squared_error", "friedman_mse","absolute_error","poisson"],
            "splitter": ["best", "random"],
            "max_depth": range(1, 10, 1),
            "min_samples_split": range(2, 10),
            "min_samples_leaf": range(1, 5),
            "max_features": ["auto", "sqrt", "log2"],
            }

            decisiontree_search = RandomizedSearchCV(self.dtmodel,
                                                 param_decision,
                                                 cv=5,
                                                 random_state=0,
                                                 n_jobs=-1)

            decisiontree_search.fit(x_train, y_train)

            best_params = decisiontree_search.best_params_

            self.dtmodel = DecisionTreeRegressor(**best_params)

            self.dtmodel.fit(x_train, y_train)

            logging.info(f"The best parameter for Decision Tree are : {best_params}. Exiting the function.")

            return self.dtmodel

        except Exception as e:
            logging.info(f"There is some error in the get_best_param_decision_tree function. The error is : {str(e)} ")

    def get_best_param_random_forest(self,x_train,y_train):

        try:
            logging.info("Entering into get best param function for Random Forest")

            param_random = {
            "n_estimators": [200,300,500,700],
            "criterion": ["squared_error", "absolute_error","poisson"],
            "max_depth": range(1, 5, 1),
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": ["True", "False"]
                        }

            randomforest_search = RandomizedSearchCV(self.rfmodel,
                                                 param_random,
                                                 cv=5,
                                                 random_state=0,
                                                 n_jobs=-1)

            randomforest_search.fit(x_train, y_train)

            best_params = randomforest_search.best_params_

            self.rfmodel = RandomForestRegressor(**best_params)

            self.rfmodel.fit(x_train, y_train)

            logging.info(f"The best parameter for Random Forest are : {best_params}. Exiting the function.")

            return self.rfmodel

        except Exception as e:
            logging.info(f"There is some error in the get_best_param_random_forest function. The error is : {str(e)} ")

    # def get_best_param_extratree(self,x_train,y_train):

    #     try:
    #         logging.info("Entering into get best param function for Extra Tree")

    #         param_extra = {
    #         "n_estimators": range(100, 1000, 100),
    #         "criterion": ["squared_error", "absolute_error"],
    #         "max_depth": range(1, 10, 1),
    #         "min_samples_split": range(2, 10),
    #         "min_samples_leaf": range(1, 5),
    #         "max_features": ["auto", "sqrt", "log2"],
    #         "bootstrap": ["True", "False"]
    #                     }

    #         extratree_search = RandomizedSearchCV(self.etmodel,
    #                                              param_extra,
    #                                              cv=5,
    #                                              random_state=0,
    #                                              n_jobs=-1)

    #         extratree_search.fit(x_train, y_train)

    #         best_params = extratree_search.best_params_

    #         self.etmodel = ExtraTreesRegressor(**best_params)

    #         self.etmodel.fit(x_train, y_train)

    #         logging.info(f"The best parameter for Extra Tree are : {best_params}. Exiting the function.")

    #         return self.etmodel

    #     except Exception as e:
    #         logging.info(f"There is some error in the get_best_param_extratree function. The error is : {str(e)} ")



    # def get_best_param_adaboost(self,x_train,y_train):

    #     try:
    #         logging.info("Entering into get best param function for Ada Boost")

    #         param_adaboost = {
    #         "base_estimator": [DecisionTreeRegressor(), RandomForestRegressor()],
    #         "n_estimators": range(100, 1000, 100),
    #         "learning_rate": list(np.arange(1.0, 10.0, 1.0)),
    #         "loss": ["linear", "square","exponential"]
    #     }

    #         adaboost_search = RandomizedSearchCV(self.admodel,
    #                                              param_adaboost,
    #                                              cv=5,
    #                                              random_state=0,
    #                                              n_jobs=-1)

    #         adaboost_search.fit(x_train, y_train)

    #         best_params = adaboost_search.best_params_

    #         self.admodel = AdaBoostRegressor(**best_params)

    #         self.admodel.fit(x_train, y_train)

    #         logging.info(f"The best parameter for Adaboost are : {best_params}. Exiting the function.")

    #         return self.admodel

    #     except Exception as e:
    #         logging.info(f"There is some error in the get_best_param_adaboost function. The error is : {str(e)} ")

    def get_best_param_gradient_boost(self,x_train,y_train):

        try:
            logging.info("Entering into get best param function for Gradient Boost")

            param_gradientboost = {
            "loss": ["squared_error", "absolute_error","huber","quantile"],
            "learning_rate": list(np.arange(1.0, 5.0, 1.0)),
            "n_estimators": range(100, 1000, 200),
            "subsample": list(np.arange(0.1, 0.5, 0.1)),
            "criterion": ["friedman_mse", "squared_error","absolute_error"],
            "max_features": ["auto", "sqrt", "log2"],
            "max_depth": range(1, 5, 1),
            "tol": [1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
            "alpha": [0.1,0.3,0.7,0.9]
        }

            gradientboost_search = RandomizedSearchCV(self.gbmodel,
                                                 param_gradientboost,
                                                 cv=5,
                                                 random_state=0,
                                                 n_jobs=-1)

            gradientboost_search.fit(x_train, y_train)

            best_params = gradientboost_search.best_params_

            self.gbmodel = GradientBoostingRegressor(**best_params)

            self.gbmodel.fit(x_train, y_train)

            logging.info(f"The best parameter for GradientBoot are : {best_params}. Exiting the function.")

            return self.gbmodel

        except Exception as e:
            logging.info(f"There is some error in the get_best_param_gradient_boost function. The error is : {str(e)} ")


    def get_best_param_xgboost(self,x_train,y_train):

        try:
            logging.info("Entering into get best param function for XG Boost")

            param_xgboost = {
            "n_estimators": range(100, 1000, 200),
            "max_depth": range(1, 5, 1),
            "learning_rate": list(np.arange(1.0, 5.0, 1.0)),
            "gamma": range(0, 10, 1),
            "subsample": list(np.arange(0.1, 0.5, 0.1)),
            "tree_method": ["auto", "exact", "approx", "hist", "gpu_hist"],
            "max_leaves": range(0, 5, 1),
            "predictor": ["auto", "cpu_predictor", "gpu_predictor"],

        }

            xgboost_search = RandomizedSearchCV(self.xgmodel,
                                                 param_xgboost,
                                                 cv=5,
                                                 random_state=0,
                                                 n_jobs=-1)

            xgboost_search.fit(x_train, y_train)

            best_params = xgboost_search.best_params_

            self.xgmodel = XGBRegressor(**best_params)

            self.xgmodel.fit(x_train, y_train)

            logging.info(f"The best parameter for XG Boost are : {best_params}. Exiting the function.")

            return self.xgmodel

        except Exception as e:
            logging.info(f"There is some error in the get_best_param_xgboost_boost function. The error is : {str(e)} ") 


    # def get_best_param_knn(self,x_train,y_train):

    #     try:
    #         logging.info("Entering into get best param function for KNN")

    #         param_knn = {
    #         "n_neighbors": range(1, 20, 3),
    #         "weights": ["uniform", "distance"],
    #         "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    #         "leaf_size": range(10, 100, 10),
    #         "p": [1, 2]
    #     }


    #         knn_search = RandomizedSearchCV(self.knnmodel,
    #                                              param_knn,
    #                                              cv=5,
    #                                              random_state=0,
    #                                              n_jobs=-1)

    #         knn_search.fit(x_train, y_train)

    #         best_params = knn_search.best_params_

    #         self.knnmodel = KNeighborsRegressor(**best_params)

    #         self.knnmodel.fit(x_train, y_train)

    #         logging.info(f"The best parameter for KNN are : {best_params}. Exiting the function.")

    #         return self.knnmodel

    #     except Exception as e:
    #         logging.info(f"There is some error in the get_best_param_knn function. The error is : {str(e)} ") 

    
    # def get_best_param_svr(self,x_train,y_train):

    #     try:
    #         logging.info("Entering into get best param function for SVM")

    #         param_svr = {
    #         "kernel": ["linear","poly","rbf","sigmoid"],
    #         "gamma": ["scale", "auto"],
    #         "C": range(1,5,1),
    #         "degree": range(1, 5, 1),
    #         "tol": [1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
    #         "epsilon":list(np.arange(0.1,0.5))
    #     }


    #         svr_search = RandomizedSearchCV(self.svmmodel,
    #                                              param_svr,
    #                                              cv=5,
    #                                              random_state=0,
    #                                              n_jobs=-1)

    #         svr_search.fit(x_train, y_train)

    #         best_params = svr_search.best_params_

    #         self.svmmodel = SVR(**best_params)

    #         self.svmmodel.fit(x_train, y_train)

    #         logging.info(f"The best parameter for SVR are : {best_params}. Exiting the function.")

    #         return self.svmmodel

    #     except Exception as e:
    #         logging.info(f"There is some error in the get_best_param_svr function. The error is : {str(e)} ")



    def get_train_adjr2score(self, x_train, y_train):

        try:
            logging.info("Entering the function of train score function")

            self.lrmodel_trained = self.get_best_param_linear_model(x_train, y_train)
            self.dtmodel_trained = self.get_best_param_decision_tree(x_train, y_train)
            self.rfmodel_trained = self.get_best_param_random_forest(x_train, y_train)
            # self.etmodel_trained = self.get_best_param_extratree(x_train, y_train)
            # self.admodel_trained = self.get_best_param_adaboost(x_train, y_train)
            # self.gbmodel_trained = self.get_best_param_gradient_boost(x_train, y_train)
            self.xgmodel_trained = self.get_best_param_xgboost(x_train, y_train)
            # # self.knnmodel_trained = self.get_best_param_knn(x_train, y_train)
            # self.svmmodel_trained = self.get_best_param_svr(x_train, y_train)

            linear_pred = self.lrmodel_trained.predict(x_train)
            decision_pred = self.dtmodel_trained.predict(x_train)
            random_pred = self.rfmodel_trained.predict(x_train)
            # gbboost_pred = self.gbmodel_trained.predict(x_train)
            xg_pred = self.xgmodel_trained.predict(x_train)
            # svr_pred = self.svmmodel_trained.predict(x_train)

            linear_score = regression_metrics.adjusted_r2(x_train,y_train,linear_pred)
            decision_score = regression_metrics.adjusted_r2(x_train,y_train,decision_pred)
            random_score = regression_metrics.adjusted_r2(x_train,y_train,random_pred)
            # gbboost_score = regression_metrics.adjusted_r2(x_train,y_train,gbboost_pred)
            xg_score = regression_metrics.adjusted_r2(x_train,y_train,xg_pred)
            # svr_score = regression_metrics.adjusted_r2(x_train,y_train,svr_pred)

            train_adjr2score = [linear_score, decision_score,random_score, xg_score]

            logging.info(f"The adjusted r2 score for each model on training set is below \n  {train_adjr2score} \n Exiting the get_train_adjr2score function")
            
        except Exception as e:
            logging.info(f"There is some error in the get_train_adjr2score function. The error is : {str(e)} ")


    def get_valid_adjr2score(self, x_valid, y_valid):

        try:
            logging.info("Entering the function of valid score function")

            print(x_valid)
            print(y_valid)

            linear_pred = self.lrmodel_trained.predict(x_valid)
            decision_pred = self.dtmodel_trained.predict(x_valid)
            random_pred = self.rfmodel_trained.predict(x_valid)
            xg_pred = self.xgmodel_trained.predict(x_valid)
            # svr_pred = self.svmmodel_trained.predict(x_valid)

            print(linear_pred)

            linear_score = regression_metrics.adjusted_r2(x_valid,y_valid,linear_pred)
            decision_score = regression_metrics.adjusted_r2(x_valid,y_valid,decision_pred)
            random_score = regression_metrics.adjusted_r2(x_valid,y_valid,random_pred)
            xg_score = regression_metrics.adjusted_r2(x_valid,y_valid,xg_pred)
            # svr_score = regression_metrics.adjusted_r2(x_valid,y_valid,svr_pred)

            print(f"the adjusted r2 score for linear regression is : {linear_score}")
            print(f"the adjusted r2 score for decision tree is : {decision_score}")

            valid_adjr2score = [linear_score, decision_score, random_score,xg_score]

            print(f"the validation list is : {valid_adjr2score}")

            best_model_dict = {
            self.lrmodel_trained: linear_score,
            self.dtmodel_trained: decision_score,
            self.rfmodel_trained: random_score,
            self.xgmodel_trained: xg_score

                             }

            print(f"the best model dictionary is : {best_model_dict}")
            # To find the best model.
            self.best_model = max(zip(best_model_dict.values(), best_model_dict.keys()))[1]

            print(f"the best model is : {self.best_model} ")
            logging.info(f"The adjusted r2 score for each model on validation set is below \n  {valid_adjr2score} \n Exiting the get_train_adjr2score function")
            
            return self.best_model

        except Exception as e:
            logging.info(f"There is some error in the get_valid_adjr2score function. The error is : {str(e)} ")


    def save_best_model(self):
        """
        This function saves the best model in the BestModel Folder
        :return:BestModel
        :rtype:Model.pkl
        """
        try:
            logging.info("Entering the function of save best model")

            modelname = self.best_model
            model_dir = "BestModel"
            os.makedirs(model_dir, exist_ok=True)
            filename = "Model.pkl"
            model_path = os.path.join(model_dir, filename)
            pickle.dump(modelname, open(model_path, "wb"))

            logging.info(f"The best model is saved at {model_path}. Exiting the module")   

        except Exception as e:
             logging.info(f"There is some error in the save_best_model function. The error is : {str(e)} ")
