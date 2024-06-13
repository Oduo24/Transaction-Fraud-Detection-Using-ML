# FRAUD DETECTION USING MACHINE LEARNING

This project is a fraud detection system that uses machine learning to identify fraudulent transactions. The project includes a Flask web application for user interaction and a machine learning model to predict fraudulent transactions.

## 1. Project Structure
The project consists of the following files:

1. model.py: Contains the code for training the machine learning models and saving the best model.
2. app.py: Contains the Flask web application code.
3. templates/index.html: The main page for transaction input.
4. templates/signup.html: The signup page for adding new users.
5. templates/result.html: Displays the result of the transaction prediction.
6. static/main.css: Custom CSS for the HTML templates.

## 2. Prepare the Data
Ensure you have the Kaggle dataset 'onlinefraud.csv' in the project directory. This dataset is used to train the model. You can get it [here](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection)

## 3. Train the Model
Run the model.py script to train the model and save it. This script will train multiple models and save the best one as best_fraud_detection_model.pkl.

```
python3 model.py
```

## 4. Run the Flask Application
Start the Flask web application by running:
```
python3 app.py
```
The application will be available at http://127.0.0.1:5000.

## 5. Usage
1. Add a User
Go to http://127.0.0.1:5000/ to add a new user. Enter the username and click submit.

2. Make a Transaction
Go to http://127.0.0.1:5000/transact to make a transaction. Fill in the payment details and submit the form. The system will predict if the transaction is fraudulent or not.

