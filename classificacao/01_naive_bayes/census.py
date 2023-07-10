# 0.4796708021127626
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

with open("./data/census.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)


model = GaussianNB()
model.fit(x_train, y_train)

predict_test = model.predict(x_test)
accuracy_score(predict_test, y_test)
