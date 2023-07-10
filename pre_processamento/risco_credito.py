import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

base = pd.read_csv("./data/risco_credito.csv")

historia_label_encoder = LabelEncoder()
divida_label_encoder = LabelEncoder()
garantias_label_encoder = LabelEncoder()
renda_label_encoder = LabelEncoder()

x = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

x[:, 0] = historia_label_encoder.fit_transform(x[:, 0])
x[:, 1] = historia_label_encoder.fit_transform(x[:, 1])
x[:, 2] = historia_label_encoder.fit_transform(x[:, 2])
x[:, 3] = historia_label_encoder.fit_transform(x[:, 3])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0
)


with open("./data/risco_credito.pkl", "wb") as f:
    pickle.dump([x_train, x_test, y_train, y_test], f)
