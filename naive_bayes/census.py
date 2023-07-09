import pandas as pd
import pickle
from sklearn.calibration import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


base = pd.read_csv("./data/census.csv")

base.head(10)

workclass_encoder = LabelEncoder()
education_encoder = LabelEncoder()
maritial_status_encoder = LabelEncoder()
occupation_encoder = LabelEncoder()
relationship_encoder = LabelEncoder()
race_encoder = LabelEncoder()
sex_encoder = LabelEncoder()
native_country_encoder = LabelEncoder()

x = base.iloc[:, 0:14].values
y = base.iloc[:, 14].values

x.shape, y.shape

x[:, 1] = workclass_encoder.fit_transform(x[:, 1])
x[:, 3] = workclass_encoder.fit_transform(x[:, 3])
x[:, 5] = workclass_encoder.fit_transform(x[:, 5])
x[:, 6] = workclass_encoder.fit_transform(x[:, 6])
x[:, 7] = workclass_encoder.fit_transform(x[:, 7])
x[:, 8] = workclass_encoder.fit_transform(x[:, 8])
x[:, 9] = workclass_encoder.fit_transform(x[:, 9])
x[:, 13] = workclass_encoder.fit_transform(x[:, 13])

scaler = StandardScaler()
x = scaler.fit_transform(x)

with open("naive_bayes/census.pkl", "wb") as f:
    pickle.dump([x, y], f)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.2, random_state=0
)

x_train.shape, y_train.shape

model = GaussianNB()
model.fit(x_train, y_train)

predict = model.predict(x_test)
accuracy_score(y_test, predict)
