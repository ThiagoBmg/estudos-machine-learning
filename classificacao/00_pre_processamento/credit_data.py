import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

base = pd.read_csv("./data/credit_data.csv")

valid_age_mean = base["age"][base["age"] >= 0].mean()
base["age"][base["age"] < 0] = valid_age_mean
base["age"].fillna(valid_age_mean, inplace=True)

x = base.iloc[:, 1:4].values
y = base.iloc[:, 4].values

# scaler = StandardScaler()
# x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0
)

x_train.shape
y_train.shape
x_test.shape
y_test.shape

with open("./data/credit_data.pkl", "wb") as f:
    pickle.dump([x_train, x_test, y_train, y_test], f)
