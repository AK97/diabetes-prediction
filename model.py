from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import uniform, randint
import pandas as pd

def auc_scorer(estimator, X, y):
    y_score = estimator.predict(X) # continuous
    return roc_auc_score(y, y_score)

def optimize_parameters(data: pd.DataFrame) -> dict:

    features = data.drop(columns=['id', 'diagnosed_diabetes']).iloc[1:].astype(float)
    target = data['diagnosed_diabetes'].iloc[1:].astype(float)

    features_norm = (features - features.min()) / (features.max() - features.min())
    print(features_norm.head())

    X = features_norm
    y = target.squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = 0.2,
        random_state = 99,
        shuffle = True
    )

    param_dist = {
        "n_estimators": randint(200, 800),
        "max_depth": randint(3, 12),
        "learning_rate": uniform(0.01, 0.2),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 5),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0, 2)
    }

    reg = XGBRegressor(
        objective = "reg:squarederror",
        random_state = 99,
        tree_method = "hist"
    )

    reg_search = RandomizedSearchCV(
        estimator = reg,
        param_distributions = param_dist,
        n_iter = 50,
        scoring = auc_scorer,
        cv = 5,
        verbose = 1,
        n_jobs = -1,
        random_state = 99,
    )

    reg_search.fit(X_train, y_train)

    print("Best CV AUC:", reg_search.best_score_)
    print("Best params:", reg_search.best_params_)

    best_reg = reg_search.best_estimator_
    y_score_test = best_reg.predict(X_test)
    test_auc = roc_auc_score(y_test, y_score_test)

    print("Test ROC-AUC:", test_auc)

    return reg_search.best_params_

def predict_with_params(training_data: pd.DataFrame, test_data: pd.DataFrame, params: dict) -> pd.Series:
    
    features = training_data.drop(columns=['id', 'diagnosed_diabetes']).iloc[1:].astype(float)
    target = training_data['diagnosed_diabetes'].iloc[1:].astype(float)

    features_norm = (features - features.min()) / (features.max() - features.min())

    X = features_norm
    y = target.squeeze()

    optimized_model = XGBRegressor(
        objective = "reg:squarederror",
        random_state = 99,
        tree_method = "hist",
        **params
    )

    optimized_model.fit(X, y)

    test_features = test_data.drop(columns=['id']).astype(float)
    test_features_norm = (test_features - features.min()) / (features.max() - features.min())

    predictions = optimized_model.predict(test_features_norm)

    return pd.Series(predictions)