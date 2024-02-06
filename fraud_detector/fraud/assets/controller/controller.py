from functools import partial

import numpy as np
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import QUrl

from fraud.assets.config.config import cfg_item
from fraud.assets.view.version.version import VersionWindow

class Controller:

    def __init__(self, view, model):
        
        """
        Initializes the Controller with a specified view and model.

        Args:
            view: The view component of the MVC architecture.
            model: The model component of the MVC architecture.
        """
        self.__view = view
        self.__model = model
        self.__version = VersionWindow()
        self.__buttons = self.__view.get_buttons()
        self.__bar_buttons = self.__view.get_bar_buttons()
        self.__combo_box = self.__view.get_combo_box()
        self.__connect_toolbar()
        self.__connect_buttons()

    def __connect_toolbar(self):

        """
        Connects the toolbar buttons to corresponding model methods.

        - Connects export template, browse template, browse test, and export predictions
          toolbar buttons to their respective methods in the model.
        """

        self.__bar_buttons["expmenu"].triggered.connect(partial(self.__model.export_data, "template", None))
        self.__bar_buttons["impmenu"].triggered.connect(partial(self.__model.browse, 'template'))
        self.__bar_buttons["imptest"].triggered.connect(partial(self.__model.browse, 'test'))
        self.__bar_buttons["exppred"].triggered.connect(partial(self.__model.export_data, "predictions", None))
        self.__bar_buttons["insmenu"].triggered.connect(lambda: QDesktopServices.openUrl(QUrl(cfg_item("app", "url"))))
        self.__bar_buttons["vermenu"].triggered.connect(partial(self.__version.show_version_window, self.__version))

    def __connect_buttons(self):

        """
        Connects push buttons to corresponding methods.

        - Connects push buttons 1, 2, and 3 to their respective methods in the controller.
        """

        self.__buttons["push_buttons1"].clicked.connect(self.__on_button1_clicked)
        self.__buttons["push_buttons2"].clicked.connect(self.__on_button2_clicked)
        self.__buttons["push_buttons3"].clicked.connect(self.__on_button3_clicked)

    def __model_selector(self):

        """
        Selects a machine learning model based on the current selection in the combo box.

        - Retrieves the current text from the combo box.
        - Iterates through available models in the configuration and matches the selected model's name.
        - Returns the code associated with the selected model.

        Returns:
            str: The code of the selected machine learning model.
        """

        current_text = self.__combo_box.currentText()
        for model in cfg_item("packages","predictors"):
            if current_text == cfg_item("packages","predictors", model, "key_quality"):
                model =  cfg_item("packages","predictors",model,"code")
                return model
    
    def __on_button1_clicked(self):

        """
        Handles the click event for push button 1.

        - Attempts to process data using the selected machine learning model.
        - Displays a message indicating the success or failure of the data processing.

        Note: This method assumes the existence of a get_data_processed method in the model.

        Raises:
            Exception: An exception is raised if an error occurs during data processing.
        """

        try:
            self.__model.get_data_processed()
            result = "All data processed correctly"
            self.__view.display_message(result, "push_buttons1")
        except:
            return None    

    def __on_button2_clicked(self):

        """
        Handles the click event for push button 2.

        - Retrieves the selected machine learning model.
        - Invokes the predictor method of the model to obtain predictions.
        - Calculates and displays the number of non-fraudulent and fraudulent transactions.

        Note: This method assumes the existence of a predictor method in the model.

        Raises:
            Exception: An exception is raised if an error occurs during prediction or result calculation.
        """

        result_correct_transactions = "Number non-fraudulent Transaction:"
        result_fraudulent_transaction = "Number Fraudulent Transaction:"
        model = self.__model_selector()
        result = self.__model.predictor(model)
        if result is not None:   
            counts = np.bincount(result, minlength=2)
            result = f"{result_correct_transactions} {counts[0]}\n{result_fraudulent_transaction} {counts[1]}"
            self.__view.display_message(result, "push_buttons2")

    def __on_button3_clicked(self):

        """
        Handles the click event for push button 3.

        - Attempts to compare the results and generate a confusion matrix.
        - Formats the confusion matrix for display.
        - Displays the formatted confusion matrix using the view's display_message method.

        Raises:
            Exception: An exception is raised if an error occurs during result comparison or formatting.
        """

        try:
            result = self.format_confusion_matrix(self.__model.compare_results())
            self.__view.display_message(result, "push_buttons3")
        except:
            return None
    

    def format_confusion_matrix(self, matrix):

        """
        Formats a confusion matrix for display.

        - Takes a confusion matrix as input.
        - Provides human-readable descriptions for each matrix element.
        - Calculates the percentage for each element in the confusion matrix.
        - Returns a formatted string containing descriptions and corresponding percentages.

        Args:
            matrix (numpy.ndarray): The confusion matrix to be formatted.

        Returns:
            str: The formatted confusion matrix as a string.
        """

        descriptions = [
            "True Negative (TN): Correctly predicted non-fraudulent transactions",
            "False Positive (FP): Incorrectly predicted fraudulent transactions",
            "False Negative (FN): Incorrectly predicted non-fraudulent transactions",
            "True Positive (TP): Correctly predicted fraudulent transactions"
        ]

        result_str = ""
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                percentage = matrix[i, j] * 100
                result_str += f"{descriptions[i * len(matrix) + j]}: {percentage:.2f}%\n"

        return result_str
    

        
    

    