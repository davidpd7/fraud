from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QDialog

from fraud.assets.config.config import cfg_item

class VersionWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("App Version Information")

        layout = QVBoxLayout()
        name = cfg_item("app", "name")
        author = cfg_item("app", "author")
        author_email = cfg_item("app", "author_email")
        version = cfg_item("app", "version")
        url = cfg_item("app", "url")

        version_label = QLabel(f"Version: {name} {version}")
        author_label = QLabel(f"Author: {author}")
        email_label = QLabel(f"Author Email: {author_email}")
        url_label = QLabel(f"URL: {url}")

        layout.addWidget(version_label)
        layout.addWidget(author_label)
        layout.addWidget(email_label)
        layout.addWidget(url_label)

        self.setLayout(layout)
    
    def show_version_window(self, version_dialog):
        version_dialog.exec()
