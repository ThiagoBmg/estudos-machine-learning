import pickle
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def census():
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

    x[:, 1] = workclass_encoder.fit_transform(x[:, 1])
    x[:, 3] = education_encoder.fit_transform(x[:, 3])
    x[:, 5] = maritial_status_encoder.fit_transform(x[:, 5])
    x[:, 6] = occupation_encoder.fit_transform(x[:, 6])
    x[:, 7] = relationship_encoder.fit_transform(x[:, 7])
    x[:, 8] = race_encoder.fit_transform(x[:, 8])
    x[:, 9] = sex_encoder.fit_transform(x[:, 9])
    x[:, 13] = native_country_encoder.fit_transform(x[:, 13])

    onehotencoder = ColumnTransformer(
        transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],
        remainder="passthrough",
    )
    x = onehotencoder.fit_transform(x).toarray()

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=0
    )

    with open("./data/census.pkl", "wb") as f:
        pickle.dump([x_train, x_test, y_train, y_test], f)


def credit_data():
    base = pd.read_csv("./data/credit_data.csv")

    valid_age_mean = base["age"][base["age"] >= 0].mean()
    base["age"][base["age"] < 0] = valid_age_mean
    base["age"].fillna(valid_age_mean, inplace=True)

    x = base.iloc[:, 1:4].values
    y = base.iloc[:, 4].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=0
    )

    x_train.shape
    y_train.shape
    x_test.shape
    y_test.shape

    with open("./data/credit_data.pkl", "wb") as f:
        pickle.dump([x_train, x_test, y_train, y_test], f)


def risco_credito():
    base = pd.read_csv("./data/risco_credito.csv")

    historia_label_encoder = LabelEncoder()
    divida_label_encoder = LabelEncoder()
    garantias_label_encoder = LabelEncoder()
    renda_label_encoder = LabelEncoder()

    x = base.iloc[:, 0:4].values
    y = base.iloc[:, 4].values

    x[:, 0] = historia_label_encoder.fit_transform(x[:, 0])
    x[:, 1] = divida_label_encoder.fit_transform(x[:, 1])
    x[:, 2] = garantias_label_encoder.fit_transform(x[:, 2])
    x[:, 3] = renda_label_encoder.fit_transform(x[:, 3])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=0
    )

    with open("./data/risco_credito.pkl", "wb") as f:
        pickle.dump([x_train, x_test, y_train, y_test], f)


if __name__ == "__main__":
    census()
    credit_data()
    risco_credito()
