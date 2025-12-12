import pandas as pd

def import_data(dataset:str) -> pd.DataFrame:

    # columns = [
    #     'id',
    #     'age',
    #     'alcohol_consumption_per_week',
    #     'physical_activity_minutes_per_week',
    #     'diet_score',
    #     'sleep_hours_per_day',
    #     'screen_time_hours_per_day',
    #     'bmi',
    #     'waist_to_hip_ratio',
    #     'systolic_bp',
    #     'diastolic_bp',
    #     'heart_rate',
    #     'cholesterol_total',
    #     'hdl_cholesterol',
    #     'ldl_cholesterol',
    #     'triglycerides',
    #     'gender',
    #     'ethnicity',
    #     'education_level',
    #     'income_level',
    #     'smoking_status',
    #     'employment_status',
    #     'family_history_diabetes',
    #     'hypertension_history',
    #     'cardiovascular_history',
    #     'diagnosed_diabetes']

    data = pd.read_csv(f'data/{dataset}.csv')
    return data

def convert_categorical_to_numerical(data: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(data, drop_first=False)