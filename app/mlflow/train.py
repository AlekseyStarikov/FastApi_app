import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pickle

# Загрузка данных
def Load_data():
    df = pd.read_csv("C:\Users\user\FastApi-app\Response_кл.csv",sep=';')
    X=df
    cus_leng_map={'less than 3 years':0,'from 3 to 7 years':1,'more than 7 years':2}
    X['cus_leng']=X['cus_leng'].map(cus_leng_map)
    X['cus_leng']=X['cus_leng'].fillna(X['cus_leng'].median())
    X['age']=X['age'].fillna(X['age'].median())
    X[['mortgage', 'life_ins','cre_card','deb_card','mob_bank','curr_acc','internet','perloan','savings','atm_user','markpl','response']] = X[['mortgage', 'life_ins','cre_card','deb_card','mob_bank','curr_acc','internet','perloan','savings','atm_user','markpl','response']].apply(lambda x: pd.factorize(x)[0])

    y=X['response']
    X=X.drop('response',axis=1)
    return X,y

X,y=Load_data()
# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)


# Настройка MLFlow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ML Models")

# Модель 1: Logistic Regression
with mlflow.start_run(run_name="XGBClassifier"):
    model = XGBClassifier(n_estimators = 59, max_depth = 5, random_state=21)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.xgboost.log_model(model, "XGB_model")

# Модель 2: Random Forest
with mlflow.start_run(run_name="RandomForest"):
    model = RandomForestClassifier(n_estimators = 59,  max_depth=5,  random_state=21)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "rf_model")

# def predict(a):
#  X,y = Load_data()
#  # Разделение данных
#  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

#  return A