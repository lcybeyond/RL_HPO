import pandas as pd
from sklearn.model_selection import train_test_split


def read_data():
    root_path = "digital_minist_data/"
    dataset = pd.read_csv(root_path + "train.csv")
    X = dataset.values[0:500, 1:]
    y = dataset.values[0:500, 0]
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=7)
    return X_train, X_test, y_train, y_test

def read_param_loss():
    dataset=pd.read_csv('./param_loss.csv',header=None)
    X=dataset.values[0:5000,0:10]
    y=dataset.values[0:5000,10]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    return X_train, X_test, y_train, y_test
