import pandas as pd
import pickle
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

base = pd.read_csv("./data/census.csv")

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

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0
)

with open("./pre_processamento/census.pkl", "wb") as f:
    pickle.dump([x_train, x_test, y_train, y_test], f)
