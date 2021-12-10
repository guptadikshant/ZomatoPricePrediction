from TrainingModels import training_models

if __name__ == "__main__":
    train_obj = training_models.TrainModel()
    train_obj.separate_train_test()
    train_obj.preprocess_train_data()
    train_obj.select_best_model()