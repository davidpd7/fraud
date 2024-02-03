import os
from importlib import resources

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMainWindow,QWidget, QGridLayout, QPushButton, QMessageBox,QComboBox
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QAction

from fraud.assets.config.config import cfg_item

class View(QMainWindow):
     
    def __init__(self):
        """
        Initializes the View class, setting up the main window with specified configurations.

        - Loads and sets the window icon.
        - Sets the window title.
        - Fixes the window size based on the specified geometry.
        - Creates a central widget and a main layout for the window.
        - Calls the __render method to perform additional setup.

        Note: Requires a valid configuration file (cfg_item) for icon path, window title,
        geometry, and any additional settings.

        Args:
            None

        Returns:
            None
        """

        super().__init__()
        icon_path = cfg_item("app","icon_path")
        with resources.path(icon_path[0], icon_path[1]) as file:
            self.__icon = file
        self.setWindowIcon(QIcon(os.path.join(self.__icon)))
        self.setWindowTitle(cfg_item("app","title"))
        self.setFixedSize(*cfg_item("app", "geometry"))
        self.__central_widget = QWidget(self)
        self.__main_layout = QGridLayout(self.__central_widget)
        self.setCentralWidget(self.__central_widget)
        self.__render()
        

    def __render(self):
   
            self.__add_menubar()
            self.__combo_box()
            self.__add_buttons()

    def __create_menu_bar(self):

        """
        Creates and configures a menu bar with specified menus and submenus.

        Returns:
            infomenu: QMenu - Menu for information-related options.
            subfileexp: QMenu - Submenu for file export options.
            subfileimp: QMenu - Submenu for file import options.
        """

        menubar = self.menuBar()
        self.__menu_bar_path = ["view", "menu_bar"]
        self.__file_menu_path = [*self.__menu_bar_path, "file_menu"]
        self.__info_menu_path = [*self.__menu_bar_path, "info_menu"]

        filemenu = menubar.addMenu(cfg_item(*self.__file_menu_path, "name"))
        infomenu = menubar.addMenu(cfg_item(*self.__info_menu_path, "name"))
        subfileimp = filemenu.addMenu(cfg_item(*self.__file_menu_path, "submenu1","name"))
        subfileexp = filemenu.addMenu(cfg_item(*self.__file_menu_path, "submenu2","name"))
        return infomenu, subfileexp, subfileimp

    def __add_menubar(self):

        """
        Adds menu bar items to the main window with specified options and actions.

        - Calls the __create_menu_bar method to obtain menu instances.
        - Creates QAction instances for various menu options.
        - Adds QAction instances to their respective submenus.
        - Populates the menu bar with information-related and file-related options.

        Note: Requires a valid configuration file (cfg_item) for menu paths, options, and any additional settings.

        Args:
            None

        Returns:
            None
        """
    
        infomenu, subfileexp, subfileimp = self.__create_menu_bar()
        impmenu = QAction(cfg_item(*self.__file_menu_path ,"submenu1","option1"), self)
        imptest = QAction(cfg_item(*self.__file_menu_path, "submenu1","option3"), self)
        expmenu = QAction(cfg_item(*self.__file_menu_path, "submenu2","option2"), self)
        exppred = QAction(cfg_item(*self.__file_menu_path, "submenu2","option4"), self)
        insmenu = QAction(cfg_item(*self.__info_menu_path,  "option1", "name"), self)
        vermenu = QAction(cfg_item(*self.__info_menu_path,  "option2", "name"), self)

        subfileimp.addAction(impmenu)
        subfileimp.addAction(imptest)
        subfileexp.addAction(expmenu)
        subfileexp.addAction(exppred)
        infomenu.addAction(insmenu)
        infomenu.addAction(vermenu)

        self.action_instances = {
            "impmenu" : impmenu,
            "expmenu" : expmenu,
            "imptest" : imptest,
            "exppred" : exppred,
            "insmenu" : insmenu,
            "vermenu" : vermenu
        }


    def __set_buttons(self, description:str ):

        """
        Creates and configures a QPushButton with the specified description.

        - Creates a QPushButton with the given description as text.
        - Sets the button size based on the configuration file.
        - Applies custom CSS styles to the button based on the configuration.
        
        Args:
            description (str): The text to be displayed on the button.

        Returns:
            button (QPushButton): The configured button instance.
        """
        
        button = QPushButton(description, parent = self.__central_widget,)
        button_size = QSize(*cfg_item("view", "buttons","button_size"))
        style = self.__css_style(cfg_item("view","buttons","style"))
        button.setFixedSize(button_size)
        button.setStyleSheet(style)
        return button

    def __add_buttons(self):
        """
        Adds QPushButton instances to the main window with specified names, positions, and descriptions.

        - Initializes an empty dictionary to store button instances.
        - Retrieves button names from the configuration file.
        - Iterates over each button name, retrieving its position and description from the configuration.
        - Calls the __set_buttons method to create and configure QPushButton instances.
        - Adds each QPushButton to the main layout at the specified position.
        - Stores button instances in the button_instances dictionary for future reference.

        Note: Requires a valid configuration file (cfg_item) for button names, positions, descriptions, and any additional settings.

        Args:
            None

        Returns:
            None
        """
        self.button_instances = {}
        button_names = cfg_item("view","buttons", "push_buttons")

        for name in button_names:
            pos = cfg_item("view","buttons", "push_buttons", name, "pos")
            description = cfg_item("view","buttons","push_buttons", name, "name")
            button = self.__set_buttons(description)
            self.__main_layout.addWidget(button, *pos)
            self.button_instances[name] = button

    def __combo_box(self):

        """
        Creates and configures a QComboBox with options from the configuration file.

        - Creates a QComboBox instance.
        - Retrieves options for the combo box from the configuration file.
        - Adds each option to the combo box.
        - Adds the combo box to the main layout.

        Note: Requires a valid configuration file (cfg_item) for combo box options and any additional settings.

        Args:
            None

        Returns:
            None
        """

        self.comboBox = QComboBox(self)
        options = cfg_item("view","combo_box")
        for item in options:
            self.comboBox.addItem(item)
        self.__main_layout.addWidget(self.comboBox)
    

    def __css_style(self, styles_data:dict):

        """
        Converts a dictionary of CSS styles into a formatted string.

        - Accepts a dictionary containing CSS style properties and their values.
        - Iterates over the key-value pairs and formats them into a CSS-style string.
        - Returns the formatted CSS-style string.

        Args:
            styles_data (dict): Dictionary containing CSS style properties and values.

        Returns:
            css_style (str): Formatted CSS-style string.
        """

        css_style = ""
        for key, value in styles_data.items():
            css_style += f"{key}: {value}; "
        return css_style


    def display_message(self, result:str, button_pressed:str):
        """
        Displays a QMessageBox with a result message based on the button pressed.

        - Determines the result message based on the button_pressed parameter.
        - Creates a QMessageBox instance, sets the text, title, icon, and adds an "Ok" button.
        - Executes the message box to display the result.

        Args:
            result: The result to be displayed in the message box.
            button_pressed (str): The identifier of the button that triggered the message.

        Returns:
            None
        """

        result_str = str(result)  

        if button_pressed in ["push_buttons1", "push_buttons2", "push_buttons3", "vermenu"]:

            pass
        else:
            return None 
        try:
            msg_box = QMessageBox()
            msg_box.setText(result_str)
            msg_box.setWindowTitle("Result")
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.addButton(QMessageBox.StandardButton.Ok)
            msg_box.exec()
        except:
            None

    def get_buttons(self):
        """
        Returns a dictionary of QPushButton instances created in the class.

        Returns:
            dict: A dictionary mapping button names to QPushButton instances.
        """
        return self.button_instances
    
    def get_bar_buttons(self):
        """
        Returns a dictionary of QAction instances related to the menu bar.

        Returns:
            dict: A dictionary mapping action names to QAction instances.
        """
        return self.action_instances

        

    def get_combo_box(self):
        """
        Returns the QComboBox instance created in the class.

        Returns:
            QComboBox: The QComboBox instance.
        """
        return self.comboBox





    


        



        
    




    

        
    


        
