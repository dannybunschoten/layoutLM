"""
PDF Information Extractor - Inference Tab
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from PIL import Image, ImageTk
import fitz  # PyMuPDF
import torch

from pdf_info_extractor.models.trainer import PDFExtractorTrainer
from pdf_info_extractor.models.inference import PDFExtractorInference


class InferenceTab:
    def __init__(self, parent, app):
        """Initialize the inference tab"""
        self.parent = parent
        self.app = app

        # Create the main frame
        self.frame = ttk.Frame(parent)

        # State variables
        self.model_path = None
        self.pdf_path = None
        self.model = None
        self.inference_engine = None
        self.results = None
        self.current_page_num = 0
        self.total_pages = 0
        self.current_image = None

        # Set up UI components
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components"""
        # Main frame
        infer_frame = ttk.Frame(self.frame)
        infer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel (settings)
        settings_frame = ttk.LabelFrame(infer_frame, text="Inference Settings")
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Model selection
        model_frame = ttk.LabelFrame(settings_frame, text="Model")
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        model_path_frame = ttk.Frame(model_frame)
        model_path_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(model_path_frame, text="Model Path:").pack(side=tk.LEFT, padx=2)
        self.model_path_var = tk.StringVar()
        ttk.Entry(model_path_frame, textvariable=self.model_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
        ttk.Button(model_path_frame, text="Browse", command=self.browse_model).pack(
            side=tk.LEFT, padx=2
        )

        self.load_model_button = ttk.Button(
            model_frame, text="Load Model", command=self.load_model
        )
        self.load_model_button.pack(fill=tk.X, padx=5, pady=2)

        # PDF selection
        pdf_frame = ttk.LabelFrame(settings_frame, text="PDF")
        pdf_frame.pack(fill=tk.X, padx=5, pady=5)

        pdf_path_frame = ttk.Frame(pdf_frame)
        pdf_path_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(pdf_path_frame, text="PDF Path:").pack(side=tk.LEFT, padx=2)
        self.pdf_path_var = tk.StringVar()
        ttk.Entry(pdf_path_frame, textvariable=self.pdf_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
        ttk.Button(pdf_path_frame, text="Browse", command=self.browse_pdf).pack(
            side=tk.LEFT, padx=2
        )

        # Page navigation
        nav_frame = ttk.Frame(pdf_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(nav_frame, text="◀", command=self.prev_page).pack(
            side=tk.LEFT, padx=2
        )
        self.page_label = ttk.Label(nav_frame, text="Page: 0/0")
        self.page_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="▶", command=self.next_page).pack(
            side=tk.LEFT, padx=2
        )

        # Inference button
        self.infer_button = ttk.Button(
            settings_frame, text="Run Inference", command=self.run_inference
        )
        self.infer_button.pack(fill=tk.X, padx=5, pady=10)

        # Output options
        output_frame = ttk.LabelFrame(settings_frame, text="Output")
        output_frame.pack(fill=tk.X, padx=5, pady=5)

        output_path_frame = ttk.Frame(output_frame)
        output_path_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(output_path_frame, text="Output File:").pack(side=tk.LEFT, padx=2)
        self.output_file_var = tk.StringVar(value="./output/results.json")
        ttk.Entry(output_path_frame, textvariable=self.output_file_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
        ttk.Button(
            output_path_frame, text="Browse", command=self.browse_output_file
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(output_frame, text="Save Results", command=self.save_results).pack(
            fill=tk.X, padx=5, pady=2
        )

        # Right panel (output view)
        results_frame = ttk.Frame(infer_frame)
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Split into document viewer and structured results
        top_frame = ttk.LabelFrame(results_frame, text="Document with Annotations")
        top_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas for viewing annotations
        self.canvas_frame = ttk.Frame(top_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(
            self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        h_scrollbar = ttk.Scrollbar(
            self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.configure(
            xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set
        )

        # Bottom panel for extracted information
        bottom_frame = ttk.LabelFrame(results_frame, text="Extracted Information")
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.extracted_info = tk.Text(bottom_frame, wrap=tk.WORD, height=10)
        self.extracted_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollbar for extracted info
        info_scrollbar = ttk.Scrollbar(
            bottom_frame, orient=tk.VERTICAL, command=self.extracted_info.yview
        )
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.extracted_info.config(yscrollcommand=info_scrollbar.set)

    def browse_model(self):
        """Browse for model directory"""
        dir_path = filedialog.askdirectory(title="Select Model Directory")

        if dir_path:
            self.model_path = dir_path
            self.model_path_var.set(dir_path)

    def browse_pdf(self):
        """Browse for PDF file"""
        file_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )

        if file_path:
            self.pdf_path = file_path
            self.pdf_path_var.set(file_path)

    def browse_output_file(self):
        """Browse for output file"""
        file_path = filedialog.asksaveasfilename(
            title="Save Results As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if file_path:
            self.output_file_var.set(file_path)

    def load_model(self):
        """Load the model"""
        self.model_path = self.model_path_var.get()

        if not self.model_path:
            messagebox.showerror("Error", "Please select a model directory")
            return

        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", "Model directory doesn't exist")
            return

        try:
            self.app.update_status("Loading model...")
            self.load_model_button.config(state=tk.DISABLED)

            # Create a thread to load the model
            threading.Thread(target=self._load_model_thread, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.app.update_status("Failed to load model")
            self.load_model_button.config(state=tk.NORMAL)

    def _load_model_thread(self):
        """Thread function to load the model"""
        try:
            # Initialize the trainer
            trainer = PDFExtractorTrainer(self.app.config, self.app.processor)

            # Load the model
            self.model = trainer.load_model(self.model_path)

            # Initialize inference engine
            self.inference_engine = PDFExtractorInference(
                self.app.config, self.app.processor, self.model
            )

            # Load label mapping if available
            label_mapping_path = os.path.join(self.model_path, "label_mapping.json")
            if os.path.exists(label_mapping_path):
                self.inference_engine.load_label_mapping(label_mapping_path)

            # Update UI from the main thread
            self.frame.after(
                0,
                lambda: [
                    self.app.update_status("Model loaded successfully"),
                    self.load_model_button.config(state=tk.NORMAL),
                ],
            )
        except Exception as e:
            # Update UI from the main thread
            self.frame.after(
                0,
                lambda: [
                    messagebox.showerror("Error", f"Failed to load model: {str(e)}"),
                    self.app.update_status("Failed to load model"),
                    self.load_model_button.config(state=tk.NORMAL),
                ],
            )

    def run_inference(self):
        """Run inference on the PDF"""
        self.pdf_path = self.pdf_path_var.get()

        if not self.model:
            messagebox.showerror("Error", "Please load a model first")
            return

        if not self.pdf_path:
            messagebox.showerror("Error", "Please select a PDF file")
            return

        if not os.path.exists(self.pdf_path):
            messagebox.showerror("Error", "PDF file doesn't exist")
            return

        try:
            self.app.update_status("Running inference...")
            self.infer_button.config(state=tk.DISABLED)

            # Create a thread to run inference
            threading.Thread(target=self._run_inference_thread, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run inference: {str(e)}")
            self.app.update_status("Inference failed")
            self.infer_button.config(state=tk.NORMAL)

    def _run_inference_thread(self):
        """Thread function to run inference"""
        try:
            # Run inference
            self.results = self.inference_engine.process_pdf(self.pdf_path)

            # Open the PDF for display
            self.doc = fitz.open(self.pdf_path)
            self.total_pages = len(self.doc)
            self.current_page_num = 0

            # Extract structured information
            structured_info = {}
            for page_num, boxes in self.results.items():
                page_info = self.inference_engine.extract_structured_info(boxes)
                for label, texts in page_info.items():
                    if label not in structured_info:
                        structured_info[label] = []
                    structured_info[label].extend(texts)

            # Update UI from the main thread
            self.frame.after(
                0,
                lambda: [
                    self.display_results(structured_info),
                    self.load_page(),
                    self.app.update_status("Inference completed"),
                    self.infer_button.config(state=tk.NORMAL),
                ],
            )
        except Exception as e:
            # Update UI from the main thread
            self.frame.after(
                0,
                lambda: [
                    messagebox.showerror("Error", f"Inference failed: {str(e)}"),
                    self.app.update_status("Inference failed"),
                    self.infer_button.config(state=tk.NORMAL),
                ],
            )

    def display_results(self, structured_info):
        """Display structured information in the text widget"""
        self.extracted_info.config(state=tk.NORMAL)
        self.extracted_info.delete(1.0, tk.END)

        for label, texts in structured_info.items():
            self.extracted_info.insert(tk.END, f"\n=== {label} ===\n", "heading")
            for text in texts:
                self.extracted_info.insert(tk.END, f"• {text}\n")

        self.extracted_info.config(state=tk.DISABLED)

        # Add tag configuration for headings
        self.extracted_info.tag_configure("heading", font=("TkDefaultFont", 10, "bold"))

    def load_page(self):
        """Load the current page with annotations"""
        if not hasattr(self, "doc"):
            return

        # Get page and render to image
        page = self.doc[self.current_page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        self.current_image = img

        # Convert to PhotoImage for display
        self.img_tk = ImageTk.PhotoImage(image=img)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, img.width, img.height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

        # Update page number display
        self.page_label.config(
            text=f"Page: {self.current_page_num + 1}/{self.total_pages}"
        )

        # Draw boxes if there are results for this page
        if self.results and self.current_page_num in self.results:
            self.draw_boxes(self.results[self.current_page_num])

    def draw_boxes(self, boxes):
        """Draw annotated boxes on the canvas"""
        for i, box in enumerate(boxes):
            x0, y0, x1, y1, text, label = box

            # Determine color based on label
            if label == "O" or label == "UNKNOWN":
                outline = "blue"
                width = 1
            else:
                outline = "green"
                width = 2

            # Draw box
            self.canvas.create_rectangle(
                x0, y0, x1, y1, outline=outline, width=width, tags=("box", f"box_{i}")
            )

            # Draw text label
            self.canvas.create_text(
                x0,
                y0 - 10,
                text=f"{label}: {text[:20]}{'...' if len(text) > 20 else ''}",
                anchor=tk.SW,
                fill=outline,
                tags=("box", f"box_{i}"),
            )

    def prev_page(self):
        """Go to previous page"""
        if hasattr(self, "doc") and self.current_page_num > 0:
            self.current_page_num -= 1
            self.load_page()

    def next_page(self):
        """Go to next page"""
        if hasattr(self, "doc") and self.current_page_num < self.total_pages - 1:
            self.current_page_num += 1
            self.load_page()

    def save_results(self):
        """Save inference results to a file"""
        if not self.results:
            messagebox.showinfo("Info", "No results to save")
            return

        output_file = self.output_file_var.get()

        if not output_file:
            messagebox.showerror("Error", "Please specify an output file")
            return

        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Create output structure
            output_data = {
                os.path.basename(self.pdf_path): {
                    str(page_num): boxes for page_num, boxes in self.results.items()
                }
            }

            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)

            self.app.update_status(f"Results saved to {output_file}")
            messagebox.showinfo("Success", f"Results saved to {output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
