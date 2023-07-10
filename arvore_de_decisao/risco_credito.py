# 0.25
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

with open("./data/risco_credito.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)


model = RandomForestClassifier(
    n_estimators=20, criterion="entropy", random_state=0
)
model.fit(x_train, y_train)

predict_test = model.predict(x_test)
accuracy_score(predict_test, y_test)
