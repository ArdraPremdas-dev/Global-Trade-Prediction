import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

data = pd.read_excel('Traindataset.xlsx')
data.head(10)
label_cols = ['Reporting Economy', 'Product/Sector']
for col in label_cols:
    data[col] = pd.factorize(data[col])[0]
data.head(10)
X = data[['Reporting Economy', 'Product/Sector', 'Year']]
y = data['Value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_regressor = XGBRegressor(random_state=42)
xgb_regressor.fit(X_train, y_train)
# Make predictions on the test set
predictions = xgb_regressor.predict(X_test)

# Calculate and print the evaluation metrics
mse = mean_squared_error(y_test, predictions)
r_squared = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R-squared:", r_squared)
predictions = xgb_regressor.predict([[0,0,2021]])
# model_filename = 'xgboost_model.pkl'
# joblib.dump(xgb_regressor, model_filename)

from flask import Flask, request, jsonify
import pandas as pd
from xgboost import XGBRegressor
import joblib


# Load the model
loaded_model = joblib.load('xgboost_model.pkl')

try:
    test_data = [0, 0, 2021]
    prediction = loaded_model.predict([test_data])[0]

        # Return the prediction as JSON
    print({'prediction': prediction})
except Exception as e:
    print({'error': str(e)})
