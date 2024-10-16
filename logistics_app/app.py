from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model = load_model('models/delay_prediction_model.h5')

# Load the scaler (using the same dummy data to fit the scaler)
scaler = StandardScaler()
dummy_data = pd.DataFrame({
    'shipping_time': [10, 12, 15, 20, 9],
    'distance': [200, 300, 150, 500, 100],
    'weather': [0, 1, 0, 2, 0],  # 0-clear, 1-rain, 2-snow
    'traffic_conditions': [0, 1, 2, 2, 0]  # 0-low, 1-medium, 2-high
})
scaler.fit(dummy_data)

# Prediction function
def predict_delay(shipping_time, distance, weather, traffic_conditions):
    # Prepare the input
    input_data = np.array([[shipping_time, distance, weather, traffic_conditions]])
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_data_scaled)
    return 'Delayed' if prediction > 0.5 else 'On Time'

# Define the main route for the web application
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get form input values
        shipping_time = float(request.form['shipping_time'])
        distance = float(request.form['distance'])
        weather = int(request.form['weather'])
        traffic_conditions = int(request.form['traffic_conditions'])

        # Make prediction
        prediction = predict_delay(shipping_time, distance, weather, traffic_conditions)

    return render_template('index.html', prediction=prediction)

# Run the Flask web server
if __name__ == "__main__":
    app.run(debug=True)
