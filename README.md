# Diabetes Prediction
Prediction attempt for the Kaggle "Playground Series - Season 5, Episode 12" competition. First we train an XGBoost model with randomized hyperparameter search and then predict on the test set, producing a submission file for diabetes diagnosis prediction.

## Strategy
- Loads `data/train.csv` and `data/test.csv` from the competition export.
- One-hot encode categorical columns with `pandas.get_dummies`, keeping all levels.
- Min-max normalizes numeric features using the training set statistics.
- Tunes an `XGBRegressor` with `RandomizedSearchCV` using ROC-AUC as the custom scoring metric.
- Trains the best model on the full training set and writes `predictions.csv` for submission.

## Requirements
- Packages: `requirements.txt`
- Kaggle data files placed in `data/train.csv` and `data/test.csv`

## Files
- `main.py` – orchestrates data loading, preprocessing, hyperparameter search, and prediction writing.
- `model.py` – XGBoost training utilities, including randomized search and inference with saved params.
- `utils.py` – CSV loader and one-hot encoder helper.

## Modeling notes
- Categorical handling: one-hot encoding without dropping any levels to avoid losing information.
- Feature scaling: min-max normalization based on the training set min/max values.
- Search space: randomized over depth, estimators, learning rate, subsample, column sample, and regularization terms for 50 trials with 5-fold CV.
- Metric: ROC-AUC calculated on continuous predictions for both CV scoring and hold-out evaluation inside the search.

## Results
- Score achieved: **0.69474**
- Highest score on leaderboard: **0.70555**