"""
PDF Information Extractor - UI Entry Point
"""

import tkinter as tk
from pdf_info_extractor.ui.app import PDFInfoExtractorApp


def main():
    """Start the UI application"""
    root = tk.Tk()
    app = PDFInfoExtractorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
