from flask import Flask, jsonify, request
from joblib import load

app = Flask(__name__)

# Load your machine learning model

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features_values and string_value from request
    data = request.json  # Assuming JSON input with 'features' and 'string_value' fields

    features_values = data.get('features', [])
    string_value = data.get('string_value', '')
    model = load(string_value + '.joblib')
    # Make prediction
    prediction = model.predict([features_values])

    # Store the features and string_value somewhere or perform additional actions

    # Return prediction as JSON response
    return jsonify({
        'prediction': prediction.tolist(),
    })


if __name__ == '__main__':
    app.run(debug=True)
