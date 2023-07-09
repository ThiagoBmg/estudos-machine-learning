import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

with open("./pre_processamento/risco_credito.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

model = GaussianNB()
model.fit(x_train, y_train)
predict = model.predict(x_test)

accuracy_score(y_test, predict)
