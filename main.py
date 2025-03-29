import tkinter as tk
from pdf_ocr_app import OCRAnnotationApp

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRAnnotationApp(root)
    root.mainloop()
