from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import tkinter as tk
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_csv():
    global data
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        info_label.config(text=f"CSV file loaded successfully: {file_path}")

def run_analysis():
    if 'data' not in globals():
        info_label.config(text="Please load a CSV file first.")
        return

    # Extract features (X) and target variable (y)
    X = data.drop("placement_package", axis=1)
    y = data["placement_package"]

    # Normalize the data using StandardScaler
    scaler = MinMaxScaler(feature_range=(0, 10))
    X_normalized = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)

    # Multiple Linear Regression
    mlr_model = LinearRegression()
    mlr_model.fit(X_train, y_train)
    y_pred_mlr = mlr_model.predict(X_test)

    # Neural Network (ANN) using TensorFlow
    ann_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000)
    ann_model.fit(X_train, y_train)
    y_pred_ann = ann_model.predict(X_test)

    # Evaluate the models
    linear_mse = mean_squared_error(y_test, y_pred_linear)
    linear_mae = mean_absolute_error(y_test, y_pred_linear)
    linear_r2 = r2_score(y_test, y_pred_linear)

    mlr_mse = mean_squared_error(y_test, y_pred_mlr)
    mlr_mae = mean_absolute_error(y_test, y_pred_mlr)
    mlr_r2 = r2_score(y_test, y_pred_mlr)

    ann_mse = mean_squared_error(y_test, y_pred_ann)
    ann_mae = mean_absolute_error(y_test, y_pred_ann)
    ann_r2 = r2_score(y_test, y_pred_ann)

    # Results
    result_text = (
        f"Linear Regression Results:\nMSE: {linear_mse}, \nMAE: {linear_mae}, \nR-squared: {linear_r2}\n\n"
        f"Multiple Linear Regression Results:\nMSE: {mlr_mse}, \nMAE: {mlr_mae}, \nR-squared: {mlr_r2}\n\n"
        f"Artificial Neural Network (ANN) Results:\nMSE: {ann_mse}, \nMAE: {ann_mae}, \nR-squared: {ann_r2}\n\n"
    )
    result_label.config(text=result_text)

    # Plotting
    plot_graph([linear_mse, mlr_mse, ann_mse], [linear_mae, mlr_mae, ann_mae], [linear_r2, mlr_r2, ann_r2])

def plot_graph(mse_values, mae_values, r2_values):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(models, mse_values, color=['blue', 'green', 'orange'])
    axes[0].set_title('Mean Squared Error (MSE)')

    axes[1].bar(models, mae_values, color=['blue', 'green', 'orange'])
    axes[1].set_title('Mean Absolute Error (MAE)')

    axes[2].bar(models, r2_values, color=['blue', 'green', 'orange'])
    axes[2].set_title('R-squared (R²)')

    plt.tight_layout()

    # Embedding the matplotlib graph in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=4, column=0, columnspan=2)

# GUI Setup
root = tk.Tk()
root.title("Placement Package Prediction Analysis")

# Load CSV Button
load_csv_button = tk.Button(root, text="Load CSV File", command=load_csv)
load_csv_button.grid(row=0, column=0, pady=10)

# Run Analysis Button
run_analysis_button = tk.Button(root, text="Run Analysis", command=run_analysis)
run_analysis_button.grid(row=0, column=1, pady=10)

# Info Label
info_label = tk.Label(root, text="Please load a CSV file.")
info_label.grid(row=1, column=0, columnspan=2)

# Results Label
result_label = tk.Label(root, text="")
result_label.grid(row=2, column=0, columnspan=2)

# Models
models = ['Linear', 'Multiple Linear', 'Neural Network']

# Start the Tkinter main loop
root.mainloop()
