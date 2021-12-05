from TrainingModels import training

if __name__ == "__main__":
    train_obj = training.TrainModel()
    train_obj.separate_train_test()
    train_obj.preprocess_train_data()