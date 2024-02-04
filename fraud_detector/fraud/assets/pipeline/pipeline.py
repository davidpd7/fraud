import pandas as pd

from fraud.assets.packages.transformers.transformers import Transformers
from fraud.assets.config.config import cfg_item

transformer = Transformers()

class Pipeline:

    """
    A class representing a data processing pipeline.
    """

    def __init__(self):
        """
        Initializes the Pipeline class by setting up configuration items and transformers.
        """
        self.__setup_configuration()
        self.__setup_transformers()

    def __setup_configuration(self):
        """
        Sets up configuration items used in the pipeline.
        """
        self._transaction_column = cfg_item("template", "columns", "datetime", "transaction")
        self._date_of_birth_column = cfg_item("template", "columns", "datetime", "date_of_birth")
        self._amount_column = cfg_item("transformed_data", "columns", "binned", "amount")
        self._transaction_age = cfg_item("transformed_data", "columns", "datetime", "transaction_age")
        self._transaction_amount_diff = cfg_item("transformed_data", "columns", "transformations", "transaction_amount_diff")
        self._time_since_last_transaction = cfg_item("transformed_data", "columns", "datetime", "time_since_last_transaction")

    def __setup_transformers(self):
        """
        Sets up transformers used in the pipeline.
        """
        self._amt_binner = transformer.get_transformer("binning_amt")
        self._tslt_binner = transformer.get_transformer("binning_tslt")
        self._meandiff_binner = transformer.get_transformer("binning_meandiff")
        
    def datetime_check(self, data:pd.DataFrame):
        """
        Checks and converts specified columns to datetime format.

        Args:
            data (pd.DataFrame): The DataFrame to perform datetime checks on.

        Returns:
            bool or None: True if datetime conversion is successful, None if an error occurs.
        """
        date_time_columns = [self._transaction_column, self._date_of_birth_column]
        try:
            for column in date_time_columns:
                data[column] = pd.to_datetime(data[column])
            return True
        except Exception as e:
            print(ValueError(f"Datetime error: Please check the following columns and make sure they follow the established format: {e}"))
            return None


    def data_transformation(self, data:pd.DataFrame):
        """
        Performs data transformations on the provided DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to undergo data transformations.

        Returns:
            pd.DataFrame: The DataFrame with added transformed columns.
        """
        try:
            mean_transaction_amount_col = cfg_item("transformed_data", "columns", "transformations", "mean_transaction_amount")
            cc_num_col = cfg_item("transformed_data", "columns", "transformations", "credit_card_number")
            transaction_hour_col = cfg_item("transformed_data", "columns", "datetime", "transaction_hour")
            transaction_day_col = cfg_item("transformed_data", "columns", "datetime", "transaction_day")
            transaction_age_col = cfg_item("transformed_data", "columns", "datetime", "transaction_age")
            number_days_year_col = cfg_item("transformed_data", "columns", "datetime", "number_days_year")
            unix_time_col = cfg_item("transformed_data", "columns", "datetime", "unix_time")

            data[mean_transaction_amount_col] = data.groupby(cc_num_col)[self._amount_column].transform("mean")
            data[self._transaction_amount_diff] = data[self._amount_column] - data[mean_transaction_amount_col]
            data[self._time_since_last_transaction] = data.groupby(cc_num_col)[unix_time_col].diff().fillna(0)
            data[transaction_hour_col] = data[self._transaction_column].dt.hour
            data[transaction_day_col] = data[self._transaction_column].dt.dayofweek
            data[transaction_age_col] = (data[self._transaction_column] - data[self._date_of_birth_column]).dt.days // number_days_year_col

            return data
        except Exception as e:
            print( ValueError(f"Error during data transformation: {e}"))
            return None

    def optimal_binning(self, data):
        """
        Applies optimal binning to specified columns in the provided DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to undergo optimal binning.

        Returns:
            pd.DataFrame or None: The DataFrame with added binned columns or None if an error occurs.
        """
        amount_binned_col = cfg_item("transformed_data", "columns", "binned", "amount_binned")
        transaction_amount_diff_binned_col = cfg_item("transformed_data", "columns", "binned", "transaction_amount_diff")
        time_since_last_transaction_binned_col = cfg_item("transformed_data", "columns", "binned", "time_since_last_transaction")
        metric = "bins"

        try:
            data[amount_binned_col] = self._amt_binner.transform(data[self._amount_column], metric=metric)
            data[time_since_last_transaction_binned_col] = self._tslt_binner.transform(data[self._time_since_last_transaction], metric=metric)
            data[transaction_amount_diff_binned_col] = self._meandiff_binner.transform(data[self._transaction_amount_diff], metric=metric)
            return data
        except Exception as e:
            print(ValueError(f"Error during optimal binning: {e}"))
            return None


    def transformer(self, data):
        """
        Applies the transformer to the specified numerical and categorical columns in the provided DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to undergo transformation.

        Returns:
            pd.DataFrame or None: The transformed DataFrame or None if an error occurs.
        """
        numerical_cols = cfg_item("transformed_data", "columns", "numerical")
        categorical_cols = cfg_item("transformed_data", "columns", "categorical")

        try:
            preprocessor = transformer.get_transformer("preprocessor")
            data_processed = preprocessor.transform(data[numerical_cols + categorical_cols])
            return data_processed
        except Exception as e:
            print(ValueError (f"Error during transformation: {e}"))
            return None
