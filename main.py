from utils import import_data, convert_categorical_to_numerical
from model import optimize_parameters, predict_with_params
import pandas as pd

training_data = import_data('train')
testing_data = import_data('test')

# qualitative_features = [
#     'gender',
#     'ethnicity',
#     'education_level',
#     'income_level',
#     'smoking_status',
#     'employment_status'
# ]

training_data = convert_categorical_to_numerical(training_data)

print(list(training_data)) # print all column names

best_params = optimize_parameters(training_data)
print("Best Parameters Found:")
print(best_params)

testing_data = convert_categorical_to_numerical(testing_data)

predictions = predict_with_params(training_data, testing_data, best_params)

output = pd.DataFrame({'id': testing_data['id'], 'diagnosed_diabetes': predictions})
output.to_csv('predictions.csv', index = False)