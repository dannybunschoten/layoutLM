"""
PDF Information Extractor - Main Application Window
"""

import tkinter as tk
from tkinter import ttk
from transformers import LayoutLMv3Processor

from pdf_info_extractor.config import Config
from pdf_info_extractor.ui.annotation_tab import AnnotationTab
from pdf_info_extractor.ui.training_tab import TrainingTab
from pdf_info_extractor.ui.inference_tab import InferenceTab


class PDFInfoExtractorApp:
    def __init__(self, root):
        """Initialize the application"""
        self.root = root
        self.root.title("PDF Information Extractor")
        self.root.geometry("1280x800")

        # Initialize config
        self.config = Config()

        # Initialize LayoutLM processor
        self.processor = self.init_processor()

        # Create UI components
        self.setup_ui()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def init_processor(self):
        """Initialize the LayoutLM processor"""
        try:
            processor = LayoutLMv3Processor.from_pretrained(
                self.config.model["model_name"], apply_ocr=False
            )
            self.update_status("LayoutLM processor initialized")
            return processor
        except Exception as e:
            from tkinter import messagebox

            messagebox.showerror("Error", f"Failed to initialize processor: {str(e)}")
            self.update_status("Failed to initialize processor")
            return None

    def setup_ui(self):
        """Set up the UI components"""
        # Create main panel with notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.annotation_tab = AnnotationTab(self.notebook, self)
        self.training_tab = TrainingTab(self.notebook, self)
        self.inference_tab = InferenceTab(self.notebook, self)

        self.notebook.add(self.annotation_tab.frame, text="Annotation")
        self.notebook.add(self.training_tab.frame, text="Training")
        self.notebook.add(self.inference_tab.frame, text="Inference")

    def update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()
