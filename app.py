from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)

uploaded_df = None
models = {}
metrics = {}
plots = {}

def plot_actual_vs_predicted(y_test, y_pred, title):
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def train_and_evaluate(df):
    global models, metrics, plots
    X = df.drop("placement_package", axis=1)
    y = df["placement_package"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    metrics['Linear Regression'] = {
        'MSE': mean_squared_error(y_test, y_pred_lin),
        'MAE': mean_absolute_error(y_test, y_pred_lin),
        'R2': r2_score(y_test, y_pred_lin)
    }
    plots['Linear Regression'] = plot_actual_vs_predicted(y_test, y_pred_lin, "Linear Regression: Actual vs Predicted")

    # Multiple Linear Regression (same as linear regression here, but just for naming)
    mul_lin_reg = LinearRegression()
    mul_lin_reg.fit(X_train, y_train)
    y_pred_mul = mul_lin_reg.predict(X_test)
    metrics['Multiple Linear Regression'] = {
        'MSE': mean_squared_error(y_test, y_pred_mul),
        'MAE': mean_absolute_error(y_test, y_pred_mul),
        'R2': r2_score(y_test, y_pred_mul)
    }
    plots['Multiple Linear Regression'] = plot_actual_vs_predicted(y_test, y_pred_mul, "Multiple Linear Regression: Actual vs Predicted")

    # ANN
    ann = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=1000, random_state=42)
    ann.fit(X_train, y_train)
    y_pred_ann = ann.predict(X_test)
    metrics['ANN'] = {
        'MSE': mean_squared_error(y_test, y_pred_ann),
        'MAE': mean_absolute_error(y_test, y_pred_ann),
        'R2': r2_score(y_test, y_pred_ann)
    }
    plots['ANN'] = plot_actual_vs_predicted(y_test, y_pred_ann, "ANN: Actual vs Predicted")

    models['Linear Regression'] = lin_reg
    models['Multiple Linear Regression'] = mul_lin_reg
    models['ANN'] = ann

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_df, metrics, plots, models
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        uploaded_df = pd.read_csv(stream)

        required_cols = ['cgpa', '12th_marks', '10th_marks', 'communication_skill',
                         'programming_skill', 'number_of_internships', 'number_of_projects',
                         'number_of_backlog', 'placement_package']
        missing_cols = [col for col in required_cols if col not in uploaded_df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns in CSV: {missing_cols}'}), 400

        train_and_evaluate(uploaded_df)
        # After training, redirect user to predict page
        return jsonify({'message': 'File uploaded and model trained successfully.'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_page')
def predict_page():
    if not metrics or not plots:
        return redirect(url_for('home'))
    return render_template('predict.html', metrics=metrics, plots=plots)

@app.route('/predict', methods=['POST'])
def predict():
    if not models:
        return jsonify({'error': 'Model not trained yet'}), 400

    data = request.json

    try:
        features = ['cgpa', '12th_marks', '10th_marks', 'communication_skill',
                    'programming_skill', 'number_of_internships', 'number_of_projects',
                    'number_of_backlog']

        input_data = [data.get(f) for f in features]

        if None in input_data:
            return jsonify({'error': 'Missing input features'}), 400

        df_input = pd.DataFrame([input_data], columns=features)

        # Here we could choose which model to use; for example, ANN:
        prediction = models['ANN'].predict(df_input)[0]

        return jsonify({'prediction': round(float(prediction), 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_sample', methods=['POST'])
def upload_sample():
    global uploaded_df, metrics, plots, models
    try:
        uploaded_df = pd.read_csv('static/sample_data.csv')
        train_and_evaluate(uploaded_df)
        return jsonify({'message': 'Sample data loaded and model trained successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
