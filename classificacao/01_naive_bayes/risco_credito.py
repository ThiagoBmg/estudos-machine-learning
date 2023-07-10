# 0.5
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

with open("./data/risco_credito.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)
