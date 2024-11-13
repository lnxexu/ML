from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/linear_regression', methods=['POST'])
def linear_regression():
    data = request.get_json()
    df = pd.DataFrame(data)
    X = df[['feature1', 'feature2', 'feature3']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return jsonify({'predictions': y_pred.tolist()})

@app.route('/naive_bayes', methods=['POST'])
def naive_bayes():
    data = request.get_json()
    df = pd.DataFrame(data)
    X = df[['transaction_amount']]
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return jsonify({'predictions': predictions.tolist()})

@app.route('/knn', methods=['POST'])
def knn():
    data = request.get_json()
    df = pd.DataFrame(data)
    X = df[['age', 'cholesterol']]
    y = df['disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return jsonify({'predictions': predictions.tolist()})

@app.route('/svm', methods=['POST'])
def svm():
    data = request.get_json()
    df = pd.DataFrame(data)
    X = df[['age', 'previous_claims']]
    y = df['claim_likelihood']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return jsonify({'predictions': predictions.tolist()})

@app.route('/decision_tree', methods=['POST'])
def decision_tree():
    data = request.get_json()
    df = pd.DataFrame(data)
    X = df[['study_hours', 'attendance_rate']]
    y = df['at_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return jsonify({'predictions': predictions.tolist()})

@app.route('/ann', methods=['POST'])
def ann():
    data = request.get_json()
    df = pd.DataFrame(data)
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values  # Target
    X_train, X_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)