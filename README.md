# Fraud Detection App

## Overview

The Fraud Detection App is a comprehensive tool designed for fraud detection tasks, providing a user-friendly interface and programmable functionality for seamless integration into Python scripts. This README provides an overview of the app's components, functionalities, and usage instructions.

## Installation

You can install the Fraud Detection App as a Python package using the following command:

pip install fraud-detection-app

## Usage

### User Interface (UI)

### Programmatically

Import Classes:

from fraud.assets.model.models import Model

Instantiate Model:

model = Model()

# Functions Documentation

## `get_data(data_path=None)`

Retrieves data from a CSV file and performs a column correctness check.

### Parameters:
- `data_path` (str, optional): Path to the CSV file. If None, uses the browsed file path.

### Returns:
- `pd.DataFrame or None`: The DataFrame containing the imported data, or None if an error occurs.

## `get_data_processed(data_path=None)`

Retrieves and processes data from a CSV file using predefined transformations.

### Parameters:
- `data_path` (str, optional): Path to the CSV file.

### Returns:
- `pd.DataFrame or None`: The processed DataFrame, or None if an error occurs during data retrieval or processing.

## `predictor(model:str, data_path=None)`

Predicts outcomes using a specified machine learning model.

### Parameters:
- `model` (str): Code representing the desired machine learning model.
- `data_path` (str, optional): Path to the CSV file.

### Returns:
- `pd.DataFrame or None`: The predictions DataFrame, or None if an error occurs during data retrieval or processing.

## `model_metrics(model:str)`

Retrieves the evaluation metrics associated with a specified machine learning model.

### Parameters:
- `model` (str): Code representing the machine learning model.

### Returns:
- `dict`: A dictionary containing evaluation metrics for the specified model.
- `None`: If the model code is not found in the configuration.

## `__test_predictions(target_column="is_fraud")`

Reads the specified target column from the test file and returns it as a pandas Series.

### Parameters:
- `target_column` (str, optional): The name of the target column. Defaults to "is_fraud".

### Returns:
- `pd.Series`: The specified target column from the test file.

## `compare_results()`

Compares test results with predicted outcomes and returns a normalized confusion matrix.

### Returns:
- `np.ndarray or None`: The normalized confusion matrix or None if testing or predictions are not available.

## `export_data(process_type:str, path=None)`

Exports data to a CSV file.

### Parameters:
- `process_type` (str): The type of process, e.g., "template" or "predictions".
- `path` (str, optional): The path where the CSV file will be saved. If not provided, the file will be saved in the current working directory.

### Returns:
- `None or str`: None if data is not available, otherwise, the path to the exported CSV file.


Use the provided methods to perform various tasks programmatically.

## Components
1. Model
The Model class handles data processing, model prediction, and result analysis. It utilizes configurations and predefined models.

2. View
The View class represents the app's graphical user interface (GUI). It includes buttons, menus, and other UI elements.

3. Controller
The Controller class acts as an intermediary between the Model and View, facilitating communication and coordinating user interactions.

## Configuration
The app utilizes a configuration file (config.json) for settings, including model paths, UI styles, and other parameters.

## Machine Learning Models
The app includes three pre-trained models: Model 1 (Gradient Boosting), Model 2 (Grid Search with Random Forest), and Model 3 (Random Forest). 

## User Interface
The UI features buttons for data processing, model prediction, and result analysis. Menus provide options for importing/exporting data, accessing instructions, and viewing the app version.

## Data Processing
The app processes data based on a template (template_fraud.csv). It validates columns, transforms datetime features...

## Machine Learning Model Descriptions

### Model 1 - RandomForestClassifier (model_1)

- **Description:** Model 1, a RandomForestClassifier, excels in overall accuracy, making it a robust choice for predicting fraud transactions.

  - **Key Quality:** **Accuracy** - Achieving a high accuracy of 99.76%, it provides reliable overall predictions.

  - **Metrics:**
    - Accuracy: 99.76%
    - ROC Score: 85.58%
    - F1 Score: 69.79%
    - Precision Score: 68.35%
    - Recall Score: 71.28%

### Model 2 - GridSearchCV (model_2)

- **Description:** Model 2, utilizing GridSearchCV, emphasizes a balanced approach between precision and recall, suitable for a variety of fraud detection scenarios.

  - **Key Quality:** **Balanced** - Balancing precision and recall to strike a middle ground for effective fraud predictions.

  - **Metrics:**
    - Accuracy: 96.40%
    - ROC Score: 93.90%
    - F1 Score: 16.38%
    - Precision Score: 8.99%
    - Recall Score: 91.38%

### Model 3 - LogisticRegression (model_3)

- **Description:** Model 3, a LogisticRegression model, prioritizes a conservative approach with a focus on minimizing false positives.

  - **Key Quality:** **Sensitivity** - Demonstrating high recall to minimize false negatives and capture potential fraud instances.

  - **Metrics:**
    - Accuracy: 91.33%
    - ROC Score: 91.05%
    - F1 Score: 7.48%
    - Precision Score: 3.90%
    - Recall Score: 90.77%


Feel free to let me know if you have any specific modifications or additions you'd like to make!
