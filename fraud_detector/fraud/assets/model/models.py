import re
import os

from PyQt6.QtWidgets import QFileDialog
import pandas as pd
from sklearn.metrics import confusion_matrix

from fraud.assets.config.config import cfg_item
from fraud.assets.pipeline.pipeline import Pipeline
from fraud.assets.packages.predictors.predictors import Predictors

pipeline = Pipeline()
predictors = Predictors()

class Model:

    def __init__(self):
        
        """
        Initializes the Model class with default configurations.

        - Loads mandatory and non-mandatory columns from the configuration file.
        - Retrieves the mandatory indicator from the configuration.
        - Initializes browsing and test flags.

        Note: Requires a valid configuration file (cfg_item) for column definitions and settings.

        Args:
            None

        Returns:
            None
        """
        self.__mandatory_columns = cfg_item("template", "columns","mandatory")
        self.__non_mandatory_columns = cfg_item("template", "columns","non_mandatory") 
        self.__mandatory_indicator = cfg_item("template", "mandatory_indicator")
        self.__browsing = False
        self.__test = False
  
    def __columns_check(self, data:pd.DataFrame):

        """
        Checks if the provided DataFrame has the required columns based on the model"s configuration.

        - Checks if any columns contain the mandatory indicator and removes it if found.
        - Validates if all mandatory and non-mandatory columns are present in the DataFrame.
        - Returns True if the columns are correct, raises ValueError otherwise.

        Note: Requires a valid configuration file (cfg_item) for column definitions and settings.

        Args:
            data (pd.DataFrame): The DataFrame to be checked for column correctness.

        Returns:
            bool: True if the DataFrame has the correct columns, raises ValueError otherwise.
        """

        if any(data.columns.str.contains(re.escape(self.__mandatory_indicator))):
            data.columns = data.columns.str.replace(self.__mandatory_indicator, "")
        if all(col in data.columns for col in self.__mandatory_columns + self.__non_mandatory_columns):
            return True
        else:
            ValueError("Please import the correct template.")
    

    def browse(self, process_type):
        
        """
        Initiates a file browsing dialog to select a file and performs export or other actions based on the process_type.
        Function only compatible with UI app.

        Parameters:
            process_type (str): The type of process, e.g., "template" or "test".

        """
        
        if process_type == "template":
            self.__browsing = True
            self.__browsed_filename, _ = QFileDialog.getOpenFileName(None, "Open Template File", "", "CSV Files (*.csv);;All Files (*)")
        elif process_type == "test":
            self.__test = True
            self.__file_test, _ = QFileDialog.getOpenFileName(None, "Open Test File", "", "CSV Files (*.csv);;All Files (*)")


    def get_data(self, data_path=None):

        """
        Retrieves data from a CSV file and performs column correctness check.

        - If browsing is enabled, uses the browsed file path; otherwise, uses the provided data_path.
        - Reads the CSV file into a DataFrame.
        - Checks if the DataFrame has the correct columns using the __columns_check method.
        - Raises ValueError if the DataFrame is empty or has incorrect columns.

        Args:
            data_path (str, optional): Path to the CSV file. If None, uses the browsed file path.

        Returns:
            pd.DataFrame or None: The DataFrame containing the imported data, or None if an error occurs.
        """

        if self.__browsing and data_path is None:
            data_path = self.__browsed_filename
        data = pd.read_csv(data_path, index_col=0)

        if self.__columns_check(data):
            if not data.empty:
                return data
            else:
                raise ValueError("The imported dataset is empty. Please fill it out with the desired information.")
        else:
            return None


    def get_data_processed(self, data_source=None, is_path=True):

        """
        Retrieves and processes data from a CSV file or DataFrame using predefined transformations.

        - If is_path is True, calls the get_data method to obtain a DataFrame from the specified CSV file.
        - If is_path is False, assumes data_source is a DataFrame.
        - If data is successfully obtained, performs datetime check, data transformation,
        optimal binning, and additional transformations using a predefined pipeline.
        - Returns the processed DataFrame.

        Args:
            data_source (str or pd.DataFrame, optional): Path to the CSV file or DataFrame.
            is_path (bool, optional): If True, treat data_source as a file path. If False, treat data_source as a DataFrame.

        Returns:
            pd.DataFrame or None: The processed DataFrame, or None if an error occurs during data retrieval or processing.
        """
        if is_path:
            data = self.get_data(data_source)
        else:
            data = data_source

        if data is None:
            return None

        if pipeline.datetime_check(data):
            data = pipeline.data_transformation(data)
            data = pipeline.optimal_binning(data)
            processed_data = pipeline.transformer(data)
            return processed_data

    def predictor(self, model:str, data_source=None, is_path=True):

        """
        Predicts outcomes using a specified machine learning model.

        - Selects the appropriate predictor based on the specified model code.
        - If is_path is True, calls the get_data method to obtain a DataFrame from the specified CSV file and processed with get_data_processed function.
        - If is_path is False, assumes data_source is a DataFrame.
        - If data processing is successful, uses the selected predictor to make predictions.
        - Returns the predictions.

        Args:
            model (str): Code representing the desired machine learning model.
            data_source (str or pd.DataFrame, optional): Path to the CSV file or DataFrame.
            is_path (bool, optional): If True, treat data_source as a file path. If False, treat data_source as a DataFrame.

        Returns:
            pd.DataFrame or None: The predictions DataFrame, or None if an error occurs during data retrieval or processing.
        """

        if model == cfg_item("packages","predictors","model_1", "code"):
            predictor =  predictors.get_model(model)
        if model == cfg_item("packages","predictors","model_2", "code"):
            predictor =  predictors.get_model(model)
        if model == cfg_item("packages","predictors","model_3", "code"):
            predictor = predictors.get_model(model)
        if is_path:
            self.__data_processed = self.get_data_processed(data_source)
        else:
            data_source = data_source
        if self.__data_processed is None:
            return None
        self.__predictions = predictor.predict(self.__data_processed)
        return self.__predictions


    def model_metrics(self, model:str):

        """
        Retrieves the evaluation metrics associated with a specified machine learning model.

        - Checks if the specified model code is present in the configuration.
        - If found, returns the metrics associated with the model.
        - Raises an exception if the model code is not found.

        Args:
            model (str): Code representing the machine learning model.

        Returns:
            dict: A dictionary containing evaluation metrics for the specified model.
            None: If the model code is not found in the configuration.
        """
            
        if model in cfg_item("packages","predictors"):
            metrics = cfg_item("packages","predictors", model, "metrics")
        else:
            raise Exception ("Invalid Model") 
        return metrics
        

    def get_test(self, target_column="is_fraud"):

        """
        Reads the specified target column from the test file and returns it as a pandas Series.

        - Reads the specified target column from the test file using pandas read_csv.
        - Returns the specified target column as a pandas Series.

        Args:
            target_column (str, optional): The name of the target column. Defaults to "is_fraud".

        Returns:
            pd.Series: The specified target column from the test file.
        """

        test = pd.read_csv(self.__file_test, index_col=0)[target_column]
        return test

    def compare_results(self):

        """
        Compares test results with predicted outcomes and returns a normalized confusion matrix.

        - If testing and predictions are enabled, reads the "is_fraud" column from the test file.
        - Computes and returns a normalized confusion matrix.

        Returns:
            np.ndarray or None: The normalized confusion matrix or None if testing or predictions are not available.
        """

        if self.__test == True and self.__predictions is not None:
            test = self.get_test()
            cm = confusion_matrix(y_true=test, y_pred=self.__predictions, normalize="true")
            return cm
    
    def export_data(self, process_type:str, path=None):

        """
        Exports data to a CSV file.

        Parameters:
            process_type (str): The type of process, e.g., "template" or "predictions".
            path (str, optional): The path where the CSV file will be saved. If not provided, the file will be saved in the current working directory.

        Returns:
            None or str: None if data is not available, otherwise, the path to the exported CSV file.
        """

        if process_type == "template":
            mandatory_columns = [i + self.__mandatory_indicator for i in self.__mandatory_columns]
            data = pd.DataFrame(columns=mandatory_columns + self.__non_mandatory_columns)

        elif process_type == "predictions":
            if self.__predictions is not None:
                data = pd.DataFrame(self.__predictions)
            else:
                return None

        filename = cfg_item(process_type, "filename")
        if path is None:
            path, _ =  QFileDialog.getSaveFileName(None, "Select File Location", "", "CSV Files (*.csv);;All Files (*)")
            if len(path) == 0:
                return None
            return data.to_csv(path, index=False)

        return data.to_csv(os.path.join(path, filename), index=False)

    





    

  
