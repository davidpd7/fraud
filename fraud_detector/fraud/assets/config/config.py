from importlib import resources
import json

def cfg_item(*items):

    """
    Retrieves a configuration item from the loaded JSON configuration data.

    - Accepts a variable number of arguments as keys to traverse the configuration data.
    - Returns the specified configuration item.

    Args:
        *items: Variable number of keys to access the configuration data.

    Returns:
        Any: The retrieved configuration item.
    """

    data = Config.instance().data

    for key in items:
        data = data[key]

    return data

class Config:

    """
    Singleton class for managing application configuration.

    - Loads configuration data from a JSON file using the importlib and resources modules.
    - Provides a singleton instance to access the configuration data.

    Note: The Config class should be instantiated only once.

    Usage:
        config_instance = Config.instance()
        value = cfg_item("section", "key")

    Raises:
        Exception: If attempting to instantiate Config more than once.
    """
    
    __config_json_path, __config_json_filename = "fraud.assets.config", "config.json"
    __instance = None

    @staticmethod
    def instance():

        """
        Returns the singleton instance of the Config class.

        Returns:
            Config: The singleton instance of the Config class.
        """
        if Config.__instance is None:
            Config()
        return Config.__instance


    def __init__(self):

        """
        Initializes the Config class.

        - Loads configuration data from the specified JSON file using the importlib and resources modules.
        """
        if Config.__instance is None:
            Config.__instance = self
        
            with resources.path(Config.__config_json_path, Config.__config_json_filename) as json_file:
                with open(json_file) as file:
                    self.data = json.load(file)
        else:
            raise Exception ("Config only can be instanciated once") 
    