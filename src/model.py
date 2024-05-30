import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

def load_model():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    clf = RandomForestClassifier()
    clf.fit(X, Y)
    
    return clf, iris

def predict(clf, df):
    prediction = clf.predict(df)
    return prediction
