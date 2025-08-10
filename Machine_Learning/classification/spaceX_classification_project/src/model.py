from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import os, json, joblib

def prepare_data(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    df['PayloadMass'] = scaler.fit_transform(df[['PayloadMass']])
    df_dummies = df[list(df.columns[16:])]
    x = df[['PayloadMass', 'Flights', 'GridFins', 'Reused', 'Legs', 'ReusedCount']]
    x = x.add(df_dummies, fill_value=0)
    y = df['Class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=1)
    return df, x_train, x_test, y_train, y_test, scaler

def save_columns(df, x_train):
    columns_model = list(df.columns)
    columns_model.remove('Class')
    columns_model = [x for x in columns_model if x in list(x_train.columns)]
    columns = {'columns': columns_model}
    with open('models/columns_model.json', 'w') as cols:
        cols.write(json.dumps(columns))

def get_model(x_train, x_test, y_train, y_test):
    parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
    svm = SVC()
    grid = GridSearchCV(svm, parameters, cv=4)
    grid.fit(x_train, y_train)
    best_estimator = grid.best_estimator_
    print("best parameters: ", grid.best_params_)
    print("accuracy for the best estimator:", round(best_estimator.score(x_test, y_test),4) * 100,'%')
    return best_estimator

def save_model(best_estimator, scaler):
    joblib.dump(best_estimator, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler_model.pkl')


if __name__ == '__main__':
    os.chdir(os.getcwd() + '\Machine_Learning\Classification\spaceX_classification_project')
    df = pd.read_csv('data/clean_data.csv')
    df, x_train, x_test, y_train, y_test, scaler = prepare_data(df)
    best_estimator = get_model(x_train, x_test, y_train, y_test)
    save_model(best_estimator, scaler)
    save_columns(df, x_train)
    print('Model trained and saved')