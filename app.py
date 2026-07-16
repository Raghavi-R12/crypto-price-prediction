from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained machine learning model
try:
    model = joblib.load('crypto_model.pkl')
except FileNotFoundError:
    print("Warning: 'crypto_model.pkl' not found. Please run train.py first to generate it.")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    
    if request.method == 'POST':
        try:
            # Extract inputs from the HTML Form
            open_p = float(request.form['open_price'])
            high_p = float(request.form['high_price'])
            low_p = float(request.form['low_price'])
            vol = float(request.form['volume'])
            
            # Check if model is loaded properly
            if model is not None:
                features = np.array([[open_p, high_p, low_p, vol]])
                prediction = model.predict(features)[0]
                prediction_text = f"Predicted Market Price: ${prediction:,.2f}"
            else:
                prediction_text = "Model is not loaded. Please train the model first."
                
        except ValueError:
            prediction_text = "Please enter valid numeric values in all fields."
            
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    # Run specifically on port 5001 to avoid macOS AirPlay conflicts
    app.run(debug=True, port=5001)
