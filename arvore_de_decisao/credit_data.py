import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as pĺx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

base = pd.read_csv("./data/data/credit_data.csv")

valid_age_mean = base["age"][base["age"] >= 0].mean()

base["age"][base["age"] < 0] = valid_age_mean

base["age"].fillna(valid_age_mean, inplace=True)

x = base.iloc[:, 1:4].values
y = base.iloc[:, 4].values


scaler = StandardScaler()
x = scaler.fit_transform(x)

# label_encoder_income = LabelEncoder()
# label_encoder_age = LabelEncoder()
# label_encoder_loan = LabelEncoder()

# x[:, 0] = label_encoder_income.fit_transform(x[:, 0])
# x[:, 1] = label_encoder_age.fit_transform(x[:, 1])
# x[:, 2] = label_encoder_loan.fit_transform(x[:, 2])

ohe_censu = ColumnTransformer(
    transformers=[("OneHot", OneHotEncoder(), [0, 1, 2])], remainder="passthrough"
)

x = ohe_censu.fit_transform(x).toarray()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0
)

# arvore_census = DecisionTreeClassifier(criterion="entropy", random_state=0)
# arvore_census.fit(x_train, y_train)

# previsoes = arvore_census.predict(x_test)
# print(accuracy_score(y_test, previsoes))

arvore_census = RandomForestClassifier(
    n_estimators=40, criterion="entropy", random_state=0
)
arvore_census.fit(x_train, y_train)

previsoes = arvore_census.predict(x_test)
print(accuracy_score(y_test, previsoes))
