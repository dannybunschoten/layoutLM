import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import PDFOCRApp

def main():
    app = QApplication(sys.argv)
    window = PDFOCRApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()