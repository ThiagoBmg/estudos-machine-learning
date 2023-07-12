import pickle
import Orange
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

### Naive Bayes

with open("./data/census.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

    model = GaussianNB()
    model.fit(x_train, y_train)

    # 0.48
    predict_test = model.predict(x_test)
    accuracy_score(predict_test, y_test)

with open("./data/credit_data.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

    model = GaussianNB()
    model.fit(x_train, y_train)

    # 0.94
    predict_test = model.predict(x_test)
    accuracy_score(predict_test, y_test)

with open("./data/risco_credito.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

    model = GaussianNB()
    model.fit(x_train, y_train)

    # 0.5
    predict_test = model.predict(x_test)
    accuracy_score(predict_test, y_test)


### Arvore de Decisão

with open("./data/census.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

    model = RandomForestClassifier(
        n_estimators=100, criterion="entropy", random_state=0
    )
    model.fit(x_train, y_train)

    # 0.85
    predict_test = model.predict(x_test)
    accuracy_score(predict_test, y_test)

with open("./data/credit_data.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

    model = RandomForestClassifier(n_estimators=40, criterion="entropy", random_state=0)
    model.fit(x_train, y_train)

    # 0.98
    predict_test = model.predict(x_test)
    accuracy_score(predict_test, y_test)

with open("./data/risco_credito.pkl", "rb") as f:
    f.name
    x_train, x_test, y_train, y_test = pickle.load(f)

    model = RandomForestClassifier(n_estimators=20, criterion="entropy", random_state=0)
    model.fit(x_train, y_train)

    # 0.25
    predict_test = model.predict(x_test)
    accuracy_score(predict_test, y_test)


###  Aprendizado por Regras

with open("./data/credit_data_regras.csv", "rb") as f:
    base = Orange.data.Table(f.name)

    base_test, base_train = Orange.evaluation.testing.sample(base, n=0.25)

    cn2 = Orange.classification.rules.CN2Learner()
    regras = cn2(base_train)

    # regras.rule_list

    # 0.97
    predicts = Orange.evaluation.testing.TestOnTestData(
        base_train, base_test, [lambda testdata: regras]
    )
    Orange.evaluation.CA(predicts)

with open("./data/risco_credito_regras.csv", "rb") as f:
    base = Orange.data.Table(f.name)

    base_test, base_train = Orange.evaluation.testing.sample(base, n=0.25)

    cn2 = Orange.classification.rules.CN2Learner()
    regras = cn2(base_train)

    # 1
    predicts = Orange.evaluation.testing.TestOnTestData(
        base_train, base_test, [lambda testdata: regras]
    )
    Orange.evaluation.CA(predicts)


### Aprendizado Baseado Em Instancias

with open("./data/census.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

    model = KNeighborsClassifier(n_neighbors=10, metric="minkowski", p=2)
    model.fit(x_train, y_train)

    # 0.83
    predict_test = model.predict(x_test)
    accuracy_score(predict_test, y_test)


with open("./data/credit_data.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

    model = KNeighborsClassifier(n_neighbors=10, metric="minkowski", p=2)
    model.fit(x_train, y_train)

    # 0.85
    predict_test = model.predict(x_test)
    accuracy_score(predict_test, y_test)


with open("./data/risco_credito.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

    model = KNeighborsClassifier(n_neighbors=10, metric="minkowski", p=2)
    model.fit(x_train, y_train)

    # 0.25
    predict_test = model.predict(x_test)
    accuracy_score(predict_test, y_test)


# Regressão logistica

with open("./data/risco_credito.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

    model = LogisticRegression(random_state=1)
    model.fit(x_train, y_train)

    # 0.5
    predict_test = model.predict(x_test)
    accuracy_score(predict_test, y_test)


with open("./data/census.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)
    
    model = LogisticRegression(random_state=1)
    model.fit(x_train, y_train)

    # 0.85
    predict_test = model.predict(x_test)
    accuracy_score(predict_test, y_test)
