from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import os
from io import StringIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize models globally (you can also load them from trained files)
linear_model = LinearRegression()
naive_bayes_model = GaussianNB()
knn_model = KNeighborsClassifier(n_neighbors=5)
svm_model = SVC(kernel='linear')
decision_tree_model = DecisionTreeClassifier()
ann_model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/linear_regression')
def linear_regression_page():
    return render_template('linear_regression.html')

@app.route('/naive_bayes')
def naive_bayes_page():
    return render_template('naive_bayes.html')

@app.route('/knn')
def knn_page():
    return render_template('knn.html')

@app.route('/svm')
def svm_page():
    return render_template('svm.html')

@app.route('/decision_tree')
def decision_tree_page():
    return render_template('decision_tree.html')

@app.route('/ann')
def ann_page():
    return render_template('ann.html')

def process_data(df, features, target):
    if not all(feature in df.columns for feature in features):
        raise KeyError(f"One or more features {features} are not in the DataFrame columns")
    if target not in df.columns:
        raise KeyError(f"Target {target} is not in the DataFrame columns")
    
    # Convert categorical features to numerical if necessary
    for feature in features:
        if df[feature].dtype == 'object':
            df[feature] = pd.Categorical(df[feature]).codes
    
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@app.route('/linear_regression', methods=['POST'])
def linear_regression():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        df = pd.read_csv(file_path)
    elif 'data' in request.form and request.form['data'] != '':
        data = request.form['data']
        df = pd.read_json(StringIO(data))
    else:
        return jsonify({'error': 'No data provided'})

    try:
        X_train, X_test, y_train, y_test = process_data(df, ['exercise_frequency', 'diet', 'air_quality'], 'weight')
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return jsonify({'predictions': y_pred.tolist()})
    except KeyError as e:
        return jsonify({'error': str(e)})

@app.route('/naive_bayes', methods=['POST'])
def naive_bayes():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        df = pd.read_csv(file_path)
    elif 'data' in request.form and request.form['data'] != '':
        data = request.form['data']
        df = pd.read_json(StringIO(data))
    else:
        return jsonify({'error': 'No data provided'})

    try:
        X_train, X_test, y_train, y_test = process_data(df, ['fever', 'cough', 'fatigue'], 'disease')
        model = GaussianNB()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return jsonify({'predictions': predictions.tolist()})
    except KeyError as e:
        return jsonify({'error': str(e)})

@app.route('/knn', methods=['POST'])
def knn():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        df = pd.read_csv(file_path)
    elif 'data' in request.form and request.form['data'] != '':
        data = request.form['data']
        df = pd.read_json(StringIO(data))
    else:
        return jsonify({'error': 'No data provided'})

    try:
        X_train, X_test, y_train, y_test = process_data(df, ['age', 'gender', 'chronic_conditions', 'medications', 'diet', 'exercise', 'smoking'], 'health_outcome')
        n_neighbors = min(5, len(X_train))  # Ensure n_neighbors is not greater than the number of training samples
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return jsonify({'predictions': predictions.tolist()})
    except KeyError as e:
        return jsonify({'error': str(e)})

@app.route('/svm', methods=['POST'])
def svm():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        df = pd.read_csv(file_path)
    elif 'data' in request.form and request.form['data'] != '':
        data = request.form['data']
        df = pd.read_json(StringIO(data))
    else:
        return jsonify({'error': 'No data provided'})

    try:
        X_train, X_test, y_train, y_test = process_data(df, ['blood_pressure', 'heart_rate', 'cholesterol', 'smoking', 'exercise_habits', 'diet'], 'risk_category')
        model = SVC()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return jsonify({'predictions': predictions.tolist()})
    except KeyError as e:
        return jsonify({'error': str(e)})

@app.route('/decision_tree', methods=['POST'])
def decision_tree():
    # Check for file upload or form data
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        
        # Ensure the upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file.save(file_path)
        df = pd.read_csv(file_path)
    elif 'data' in request.form and request.form['data'] != '':
        data = request.form['data']
        df = pd.read_json(StringIO(data))
    else:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Generate recommendations based on the input data
        recommendations = []
        
        for index, row in df.iterrows():
            # Basic logic for generating recommendations
            if row.get('symptoms') == "fever":
                recommendations.append({
                    "morning": "Rest and stay hydrated.",
                    "diet": "Light meals, avoid heavy foods.",
                    "doctor_visit": "Consider visiting a doctor."
                })
            elif row.get('symptoms') == "cough":
                recommendations.append({
                    "morning": "30 minutes of light walking.",
                    "diet": "Warm soups and plenty of fluids.",
                    "doctor_visit": "Consult a doctor if symptoms persist."
                })
            else:
                recommendations.append({
                    "morning": "Continue your current routine.",
                    "diet": "Maintain a balanced diet with regular meals.",
                    "doctor_visit": "No immediate action required."
                })

        return jsonify({'recommendations': recommendations}), 200

    except KeyError as e:
        return jsonify({'error': f'Missing key: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    
@app.route('/ann', methods=['POST'])
def ann():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        df = pd.read_csv(file_path)
    elif 'data' in request.form and request.form['data'] != '':
        data = request.form['data']
        df = pd.read_json(StringIO(data))
    else:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Process data for training and testing
        X_train, X_test, y_train, y_test = process_data(df, 
            ['heart_rate', 'sleep_cycles', 'physical_activity_levels', 'blood_sugar_levels'], 
            'health_event'
        )
        
        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Initialize and train the ANN model
        model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Assuming we classify based on a threshold (e.g., 0.5 for binary classification)
        # Here we will assume that predictions are continuous values from 0 to 1
        # You may need to adjust this based on your actual use case
        prediction_probabilities = predictions.tolist()
        
        # Generate recommendations based on predicted probabilities
        recommendations = []
        for prob in prediction_probabilities:
            if prob >= 0.5:  # Assuming 0.5 is the threshold for a significant health event
                recommendations.append({
                    "probability": prob * 100,  # Convert to percentage
                    "recommendation": "Take rest days and hydrate."
                })
            else:
                recommendations.append({
                    "probability": prob * 100,
                    "recommendation": "Continue your current activities."
                })

        return jsonify({'prediction': recommendations}), 200

    except KeyError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)