# Fraud Detection App

## Overview

The Fraud Detection App is a comprehensive tool designed for fraud detection tasks, providing a user-friendly interface and programmable functionality for integration into Python scripts. This README provides an overview of the app's components, functionalities, and usage instructions.

Please note that this app is part of a project for a master's degree program and is only workable with the following dataset: Fraud Detection Dataset : 'https://www.kaggle.com/datasets/kartik2112/fraud-detection'

## Installation

This app is not in pipy repository, therefore the installation shall be locally. You can install the Fraud Detection App as a Python package using the following command:

pip install **`path_package_in_local_machine`**.

## Usage

This app allows the user with two different ways to use it. For one side, a friendly UI, as well as via python packages. Both alternatives contain the same functionalies. See below instructions for each option.

### Python Package

Import Classes:

from fraud.assets.model.models import Model

Instantiate Model:

model = Model()

#### `model.get_data(data_path=None)`

Retrieves data from a CSV file and performs a column correctness check.

##### Parameters:
- `data_path` (str, optional): Path to the CSV file. If None, uses the browsed file path.

##### Returns:
- `pd.DataFrame or None`: The DataFrame containing the imported data, or None if an error occurs.

#### `get_data_processed(data_source=None, is_path=True)`

Retrieves and processes data from a CSV file or DataFrame using predefined transformations.

##### Parameters:
- `data_source` (str or pd.DataFrame, optional): Path to the CSV file or DataFrame.
- is_path (bool, optional): If True, treat data_source as a file path. If False, treat data_source as a DataFrame.

##### Returns:
- `pd.DataFrame or None`: The processed DataFrame, or None if an error occurs during data retrieval or processing.

#### `predictor(model:str, data_source=None, is_path=True)`

Predicts outcomes using a specified machine learning model.

##### Parameters:
- `model` (str): Code representing the desired machine learning model.
- `data_source` (str or pd.DataFrame, optional): Path to the CSV file or DataFrame.
- `is_path` (bool, optional): If True, treat data_source as a file path. If False, treat data_source as a DataFrame.

##### Returns:
- `pd.DataFrame or None`: The predictions DataFrame, or None if an error occurs during data retrieval or processing.

#### `model.model_metrics(model:str)`

Retrieves the evaluation metrics associated with a specified machine learning model.

##### Parameters:
- `model` (str): Code representing the machine learning model.

##### Returns:
- `dict`: A dictionary containing evaluation metrics for the specified model.
- `None`: If the model code is not found in the configuration.

#### `model.test_predictions(target_column="is_fraud")`

Reads the specified target column from the test file and returns it as a pandas Series.

##### Parameters:
- `target_column` (str, optional): The name of the target column. Defaults to "is_fraud".

##### Returns:
- `pd.Series`: The specified target column from the test file.

#### `model.compare_results()`

Compares test results with predicted outcomes and returns a normalized confusion matrix.

##### Returns:
- `np.ndarray or None`: The normalized confusion matrix or None if testing or predictions are not available.

#### `model.export_data(process_type:str, path=None)`

Exports data to a CSV file.

##### Parameters:
- `process_type` (str): The type of process, e.g., "template" or "predictions".
- `path` (str, optional): The path where the CSV file will be saved. If not provided, the file will be saved in the current working directory.

##### Returns:
- `None or str`: None if data is not available, otherwise, the path to the exported CSV file.


## User Interface
The UI features buttons for data processing, model prediction, and result analysis. Menus provide options for importing/exporting data, accessing instructions, and viewing the app version.

## Data Processing
The app processes data based on a template (template_fraud.csv). It validates columns, transforms datetime features...

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
The app includes three pre-trained models: Model 1 (Logistic Regression), Model 2 (Grid Search with Random Forest), and Model 3 (Random Forest). 

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
