import joblib
from importlib import resources

from fraud.assets.config.config import cfg_item

class Predictors:

    """
    A class for loading and accessing machine learning models from packages.
    """

    _root_dir = ["packages", "predictors"]

    def __init__(self):

        """
        Initializes the Predictors class by loading packages.
        """

        self._load_packages()

    def _load_packages(self):

        """
        Loads machine learning models from packages specified in the configuration file.
        """
        
        for package_name, package_info in cfg_item(*Predictors._root_dir).items():
            package_path = package_info["path"]
            with resources.path(package_path[0], package_path[1]) as package_file:
                setattr(self, package_name, joblib.load(package_file))

    def get_model(self, model_name:str):

        """
        Returns the specified machine learning model from the loaded packages.

        Args:
            model_name (str): The name of the machine learning model.

        Returns:
            object: The specified machine learning model.
        """
        return getattr(self, model_name, None)