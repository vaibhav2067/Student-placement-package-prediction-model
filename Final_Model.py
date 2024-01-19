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
    X1 = data.drop("placement_package", axis=1)
    y1 = data["placement_package"]

    # Normalize the data using StandardScaler
    scaler = MinMaxScaler(feature_range=(0, 10))
    X_normalized = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.4, random_state=30)

    # Linear Regression - Iteration 1
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    linear_mse = mean_squared_error(y_test, y_pred_linear)
    linear_mae = mean_absolute_error(y_test, y_pred_linear)
    linear_r2 = r2_score(y_test, y_pred_linear)

    # Linear Regression - Iteration 2
    linear_model_iter2 = LinearRegression()
    linear_model_iter2.fit(X_train1, y_train1)
    y_pred_linear_iter2 = linear_model_iter2.predict(X_test1)
    linear_mse_iter2 = mean_squared_error(y_test1, y_pred_linear_iter2)
    linear_mae_iter2 = mean_absolute_error(y_test1, y_pred_linear_iter2)
    linear_r2_iter2 = r2_score(y_test1, y_pred_linear_iter2)

    # Multiple Linear Regression - Iteration 1
    mlr_model = LinearRegression()
    mlr_model.fit(X_train, y_train)
    y_pred_mlr = mlr_model.predict(X_test)
    mlr_mse = mean_squared_error(y_test, y_pred_mlr)
    mlr_mae = mean_absolute_error(y_test, y_pred_mlr)
    mlr_r2 = r2_score(y_test, y_pred_mlr)

    # Multiple Linear Regression - Iteration 2
    mlr_model_iter2 = LinearRegression()
    mlr_model_iter2.fit(X_train1, y_train1)
    y_pred_mlr_iter2 = mlr_model_iter2.predict(X_test1)
    mlr_mse_iter2 = mean_squared_error(y_test1, y_pred_mlr_iter2)
    mlr_mae_iter2 = mean_absolute_error(y_test1, y_pred_mlr_iter2)
    mlr_r2_iter2 = r2_score(y_test1, y_pred_mlr_iter2)

    # Neural Network (ANN) - Iteration 1
    ann_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=5000)
    ann_model.fit(X_train, y_train)
    y_pred_ann = ann_model.predict(X_test)
    ann_mse = mean_squared_error(y_test, y_pred_ann)
    ann_mae = mean_absolute_error(y_test, y_pred_ann)
    ann_r2 = r2_score(y_test, y_pred_ann)

    # Neural Network (ANN) - Iteration 2
    ann_model_iter2 = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=6000)
    ann_model_iter2.fit(X_train, y_train)
    y_pred_ann_iter2 = ann_model_iter2.predict(X_test)
    ann_mse_iter2 = mean_squared_error(y_test, y_pred_ann_iter2)
    ann_mae_iter2 = mean_absolute_error(y_test, y_pred_ann_iter2)
    ann_r2_iter2 = r2_score(y_test, y_pred_ann_iter2)

    # Results for Iteration 1
    result_text_iteration1 = (
        f"Linear Regression Results - Iteration 1:\nMSE: {linear_mse}, \nMAE: {linear_mae}, \nR-squared: {linear_r2}\n\n"
        f"Multiple Linear Regression Results - Iteration 1:\nMSE: {mlr_mse}, \nMAE: {mlr_mae}, \nR-squared: {mlr_r2}\n\n"
        f"Artificial Neural Network (ANN) Results - Iteration 1:\nMSE: {ann_mse}, \nMAE: {ann_mae}, \nR-squared: {ann_r2}\n\n"
    )

    # Results for Iteration 2
    result_text_iteration2 = (
        f"Linear Regression Results - Iteration 2:\nMSE: {linear_mse_iter2}, \nMAE: {linear_mae_iter2}, \nR-squared: {linear_r2_iter2}\n\n"
        f"Multiple Linear Regression Results - Iteration 2:\nMSE: {mlr_mse_iter2}, \nMAE: {mlr_mae_iter2}, \nR-squared: {mlr_r2_iter2}\n\n"
        f"Artificial Neural Network (ANN) Results - Iteration 2:\nMSE: {ann_mse_iter2}, \nMAE: {ann_mae_iter2}, \nR-squared: {ann_r2_iter2}\n\n"
    )

    result_label_iteration1.config(text=result_text_iteration1)
    result_label_iteration2.config(text=result_text_iteration2)

    # Create frames for each set of graphs
    mse_frame = tk.Frame(canvas_frame)
    mse_frame.grid(row=3, column=0, columnspan=3)

    mae_frame = tk.Frame(canvas_frame)
    mae_frame.grid(row=4, column=0, columnspan=3)

    r2_frame = tk.Frame(canvas_frame)
    r2_frame.grid(row=5, column=0, columnspan=3)

    # Plotting for MSE
    fig_mse, axes_mse = plot_graph([linear_mse, mlr_mse, ann_mse], [linear_mse_iter2, mlr_mse_iter2, ann_mse_iter2], models, 'MSE')
    canvas_mse = FigureCanvasTkAgg(fig_mse, master=mse_frame)
    canvas_widget_mse = canvas_mse.get_tk_widget()
    canvas_widget_mse.grid(row=0, column=0, columnspan=3)

    # Plotting for MAE
    fig_mae, axes_mae = plot_graph([linear_mae, mlr_mae, ann_mae], [linear_mae_iter2, mlr_mae_iter2, ann_mae_iter2], models, 'MAE')
    canvas_mae = FigureCanvasTkAgg(fig_mae, master=mae_frame)
    canvas_widget_mae = canvas_mae.get_tk_widget()
    canvas_widget_mae.grid(row=0, column=0, columnspan=3)

    # Plotting for R-squared
    fig_r2, axes_r2 = plot_graph([linear_r2, mlr_r2, ann_r2], [linear_r2_iter2, mlr_r2_iter2, ann_r2_iter2], models, 'R-squared')
    canvas_r2 = FigureCanvasTkAgg(fig_r2, master=r2_frame)
    canvas_widget_r2 = canvas_r2.get_tk_widget()
    canvas_widget_r2.grid(row=0, column=0, columnspan=3)

def plot_graph(iteration1_values, iteration2_values, models, metric_name):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 1.7))

    bar_width = 0.35  # Width of the bars
    index = range(len(models))

    # Bar plot for Iteration 1
    axes[0].bar(index, iteration1_values, width=bar_width, color='blue', label='Iteration 1')
    axes[0].set_title(f'{metric_name} - Iteration 1')
    axes[0].set_xticks(index)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(-1, max(max(iteration1_values), max(iteration2_values)) + 1)

    # Bar plot for Iteration 2
    axes[1].bar(index, iteration2_values, width=bar_width, color='green', label='Iteration 2')
    axes[1].set_title(f'{metric_name} - Iteration 2')
    axes[1].set_xticks(index)
    axes[1].set_xticklabels(models)
    axes[1].set_ylim(-1, max(max(iteration1_values), max(iteration2_values)) + 1)

    # Difference between Iteration 1 and Iteration 2
    difference_values = [i - j for i, j in zip(iteration1_values, iteration2_values)]
    axes[2].bar(index, difference_values, width=bar_width, color='orange', label='Difference (Iteration 1 - Iteration 2)')
    axes[2].set_title(f'Difference in {metric_name} between Iteration 1 and Iteration 2')
    axes[2].set_xticks(index)
    axes[2].set_xticklabels(models)

    plt.tight_layout()

    # Create a FigureCanvasTkAgg with the figure
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=6, column=0, columnspan=3)

    return fig, axes

# GUI Setup
root = tk.Tk()
root.title("Placement Package Prediction Analysis")

# Create a Canvas widget for scrolling
canvas_frame = tk.Frame(root)
canvas_frame.grid(row=0, column=0, sticky=tk.NSEW)

# Load CSV Button
load_csv_button = tk.Button(canvas_frame, text="Load CSV File", bg='#07004C', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa',command=load_csv)
load_csv_button.grid(row=0, column=0, pady=10, sticky='ew', columnspan=1)  # Span 2 columns

# Run Analysis Button
run_analysis_button = tk.Button(canvas_frame, text="Run Analysis",bg='#07004C', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa', command=run_analysis)
run_analysis_button.grid(row=0, column=2, pady=10, sticky='ew', columnspan=2)  # Span 2 columns

# Info Label
info_label = tk.Label(canvas_frame, text="Please load a CSV file.")
info_label.grid(row=1, column=0, columnspan=4, pady=5, sticky='ew')  # Span 4 columns

# Results Label - Iteration 1
result_label_iteration1 = tk.Label(canvas_frame, text="")
result_label_iteration1.grid(row=2, column=0, columnspan=3, pady=5)  # Span 4 columns

# Results Label - Iteration 2
result_label_iteration2 = tk.Label(canvas_frame, text="")
result_label_iteration2.grid(row=2, column=1, columnspan=4, pady=5)  # Span 4 columns

# Models
models = ['Linear', 'Multiple Linear', 'Neural Network']

# Start the Tkinter main loop
root.mainloop()
