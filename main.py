import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)
logger.setLevel(level="INFO")

files = [
    "./data/risco_credito.pkl",
    "./data/credit_data.pkl",
    "./data/census.pkl",
]

for file in files:
    with open(file, "rb") as f:
        x_train, x_test, y_train, y_test = pickle.load(f)

    nb_model = GaussianNB()
    nb_model.fit(x_train, y_train)

    predict_nb_model = nb_model.predict(x_test)
    print(
        {
            "file": file,
            "model": GaussianNB,
            "score": accuracy_score(predict_nb_model, y_test),
        }
    )

    dt_model = DecisionTreeClassifier()
    dt_model.fit(x_train, y_train)

    predict_dt_model = dt_model.predict(x_test)
    print(
        {
            "file": file,
            "model": DecisionTreeClassifier,
            "score": accuracy_score(predict_dt_model, y_test),
        }
    )

    rf_model = RandomForestClassifier(
        n_estimators=40, criterion="entropy", random_state=0
    )
    rf_model.fit(x_train, y_train)

    predict_rf_model = rf_model.predict(x_test)
    print(
        {
            "file": file,
            "model": RandomForestClassifier,
            "score": accuracy_score(predict_rf_model, y_test),
        }
    )
