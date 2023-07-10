# 0.8301191499815748

import pickle
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

with open("./data/census.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

model = KNeighborsClassifier(n_neighbors=10, metric="minkowski", p=2)
model.fit(x_train, y_train)

predict_test = model.predict(x_test)
accuracy_score(predict_test, y_test)
