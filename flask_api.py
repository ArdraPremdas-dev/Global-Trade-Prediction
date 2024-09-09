from flask import Flask, request, jsonify
import pandas as pd
from xgboost import XGBRegressor
import joblib

app = Flask(__name__)

# Load the model
loaded_model = joblib.load('xgb_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()

    # Make predictions
    try:
        test_data = [data['selected_col1'], data['selected_col2']]
        prediction = loaded_model.predict([test_data])[0]

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)