from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import joblib
import random

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Mock database for users
# Generate unique ids
start = 1090000000
end = 1410000000

users = {}

@app.route('/transact')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
def add_user():
    if request.method == 'POST':
        # Capture the form data
        user = request.form['user']
        if user in users:
            return jsonify({"message": "User already exists"}), 400
        # Create account number
        account_number = random.randint(start, end)
        users[user] = account_number
        print(users)
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/predict', methods=['POST'])
def predict():
    type = int(request.form['payment_type'])
    amount = 9000.60
    oldbalanceOrg = 9000.60
    newbalanceOrig = 0.0
    transaction_data = [type, amount, oldbalanceOrg, newbalanceOrig]

    transaction_features = np.array([transaction_data])
    prediction = model.predict(transaction_features)
    if prediction[0] == 'Fraud':
        return render_template('result.html', result='error', message='Transaction has been flagged as fraudulent')
    return render_template('result.html', result='success', message='Transaction is Successful')

if __name__ == '__main__':
    app.run(debug=True)