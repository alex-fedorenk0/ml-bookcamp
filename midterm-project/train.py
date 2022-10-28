import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import bentoml

df = pd.read_csv('student-por.csv', sep=';')

dv = DictVectorizer(sparse=False)
scaler = StandardScaler()

df_train, df_test, y_train, y_test = train_test_split(
    df.drop(['G1', 'G2', 'G3'], axis=1),
    df['G3'],
    test_size=0.2,
    random_state=100)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

df_train_dicts = df_train.to_dict(orient='records')
df_test_dicts = df_test.to_dict(orient='records')

X_train = dv.fit_transform(df_train_dicts)
X_test = dv.transform(df_test_dicts)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestRegressor(
    min_samples_leaf=0.01,
    n_estimators=50,
    max_depth=50,
    random_state=100)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'Model RMSE on test set: {rmse:.2f}')


bentoml.sklearn.save_model(
    'student_performance_model',
    rf,
    custom_objects={
        'dict_vectorizer': dv,
        'standard_scaler': scaler,
    }
)

print('Model saved')