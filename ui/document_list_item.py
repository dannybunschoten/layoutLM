from PyQt5.QtWidgets import QListWidgetItem
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize

class DocumentListItem(QListWidgetItem):
    """Custom list widget item for documents"""

    def __init__(self, document):
        super().__init__()
        self.document = document
        self.setText(document.filename)
        if document.thumbnail:
            self.setIcon(QIcon(document.thumbnail))
        self.setSizeHint(QSize(180, 100))