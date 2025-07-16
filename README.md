# Student Placement Package Prediction

This project is a web-based Machine Learning application that predicts a student's expected placement package based on academic and skill-related input data. Built with Flask, scikit-learn, and basic HTML/CSS/JS frontend, the application allows users to upload CSV data, view model evaluation metrics, and make real-time predictions.

## Features

- **CSV Upload:** Upload your dataset to train models.
- **Sample Dataset:** Load a pre-provided sample CSV.
- **Model Training:** Trains and evaluates three models:
  - Linear Regression
  - Multiple Linear Regression
  - Artificial Neural Network (ANN) via MLPRegressor
- **Metrics Visualization:** View MSE, MAE, R² Score with charts.
- **Placement Package Prediction:** Predict using trained ANN model.

---

## Technologies Used

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn
- **Visualization:** Matplotlib (Base64 encoded charts)
- **Frontend:** HTML, Bootstrap, JavaScript

---

## Setup Instructions

### Prerequisites

- Python 3.7+
- pip

### Install Dependencies

```bash
pip install flask pandas scikit-learn matplotlib
```

### Folder Structure

```plaintext
project/
├── app.py
├── templates/
│   ├── index.html
│   └── predict.html
├── static/
│   └── sample_data.csv  # optional
├── upload.js            # optional (used in advanced frontend)
```

### Run the Application

```bash
python app.py
```

The app will be accessible at `http://127.0.0.1:5000/`

---

## Usage

1. **Homepage:**
   - Upload your CSV file or use sample data.

2. **Dataset Requirements:**
   CSV must include these columns:

   - `cgpa`
   - `12th_marks`
   - `10th_marks`
   - `communication_skill`
   - `programming_skill`
   - `number_of_internships`
   - `number_of_projects`
   - `number_of_backlog`
   - `placement_package`

3. **Model Evaluation Page:**
   - View MSE, MAE, R² for each model.
   - Charts of actual vs. predicted values.

4. **Prediction Form:**
   - Fill the form with your academic and skill details.
   - Get predicted placement package in LPA.

---

## API Endpoints

| Route            | Method | Description                               |
|------------------|--------|-------------------------------------------|
| `/`              | GET    | Home page (CSV upload UI)                 |
| `/upload`        | POST   | Upload CSV and trigger training           |
| `/upload_sample` | POST   | Load & train using sample CSV             |
| `/predict_page`  | GET    | Shows metrics, plots, prediction form     |
| `/predict`       | POST   | API to predict placement package (JSON)   |

---

## Author

- [Your Name / GitHub](https://github.com/vaibhav2067)

---

## License

This project is licensed under the MIT License.

---

## Note

This is a prototype tool and should not be used for real-world decision-making without further validation and enhancement.

---

## Screenshot

![Screenshot](static/screenshot.png)  <!-- Replace with actual screenshot path -->
