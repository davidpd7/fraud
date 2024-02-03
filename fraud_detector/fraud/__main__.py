import sys

from PyQt6.QtWidgets import QApplication

from fraud.assets.view.view import View
from fraud.assets.controller.controller import Controller
from fraud.assets.model.models import Model

def main(args = None):

    app = QApplication(sys.argv)
    view = View()
    view.show()
    model = Model()
    controller = Controller(view, model)

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
    
    