import joblib
from importlib import resources

from fraud.assets.config.config import cfg_item
import joblib

class Transformers:

    """
    A class for loading and accessing data transformers from packages.
    """

    _root_dir = ["packages", "transformers"]

    def __init__(self):

        """
        Initializes the Transformers class by loading packages.
        """

        self._load_packages()

    def _load_packages(self):

        """
        Loads data transformers from packages specified in the configuration file.
        """

        for package_name, package_info in cfg_item(*Transformers._root_dir).items():
            package_path = package_info["path"]
            with resources.path(package_path[0], package_path[1]) as package_file:
                setattr(self, package_name, joblib.load(package_file))

    def get_transformer(self, transformer_name:str):

        """
        Returns the specified data transformer from the loaded packages.

        Args:
            transformer_name (str): The name of the data transformer.

        Returns:
            object: The specified data transformer.
        """
        return getattr(self, transformer_name, None)
