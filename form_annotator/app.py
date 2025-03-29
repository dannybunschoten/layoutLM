import os
import sys
import json
import torch
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import fitz  # PyMuPDF
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3FeatureExtractor,
)
import pytesseract
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import cv2
import logging
import threading
import queue
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("form_annotator.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Constants
FIELD_TYPES = [
    "text",
    "checkbox",
    "radio",
    "dropdown",
    "signature",
    "date",
    "number",
    "currency",
]
MODEL_PATH = "layoutlm_form_model"
DEFAULT_SCALE = 1.5


@dataclass
class FormFieldAnnotation:
    """Data class for form field annotations."""

    name: str
    field_type: str
    bbox: List[int]  # Normalized coordinates [x1, y1, x2, y2] in 0-1000 range
    page: int
    display: Dict[str, float]  # Display coordinates for canvas rendering
    label: Optional[str] = None  # For training data, the actual label/value


class LayoutLMDataset(Dataset):
    """Dataset for training/fine-tuning LayoutLMv3 with built-in OCR."""

    def __init__(self, annotations, processor, max_length=512):
        self.annotations = annotations
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]

        # We need to modify how we align OCR-extracted words with our label annotations
        # For this, we'll need a more complex approach in the prepare_training_data function

        # For OCR-based approach, we only pass the image and the labels
        # The processor will handle OCR automatically
        encoding = self.processor(
            images=item["image"],
            word_labels=item["labels"] if "labels" in item else None,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove batch dimension
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.squeeze(0)

        return encoding


class ModelThread(threading.Thread):
    """Thread class for running model inference with built-in OCR without blocking the UI."""

    def __init__(self, queue, result_queue, processor, model):
        threading.Thread.__init__(self)
        self.daemon = True  # Thread will close when main program exits
        self.queue = queue
        self.result_queue = result_queue
        self.processor = processor
        self.model = model

    def run(self):
        while True:
            try:
                # Get task from queue
                task = self.queue.get(block=True, timeout=1)

                if task["type"] == "inference":
                    # Run inference
                    image = task["image"]

                    # Process only the image - OCR will be done automatically
                    encoding = self.processor(
                        images=image,
                        truncation=True,
                        padding="max_length",
                        max_length=512,
                        return_tensors="pt",
                    )

                    with torch.no_grad():
                        outputs = self.model(**encoding)
                        predictions = outputs.logits.argmax(-1).squeeze().tolist()

                    # Get words and boxes from processor output
                    # This extracts what the OCR generated
                    words = self.processor.tokenizer.convert_ids_to_tokens(
                        encoding["input_ids"][0], skip_special_tokens=True
                    )

                    # Get the normalized boxes too (converting from tokenizer to word level)
                    # Note: This is a simplification - you might need to process these boxes
                    # to get proper word-level boxes
                    boxes = encoding["bbox"][0].tolist()

                    # Put result in result queue
                    self.result_queue.put(
                        {
                            "id": task["id"],
                            "predictions": predictions,
                            "words": words,
                            "boxes": boxes,
                        }
                    )

                elif task["type"] == "training":
                    # Run training step with OCR
                    # This would be implemented for fine-tuning
                    logger.info("Training not implemented yet")

                # Mark task as done
                self.queue.task_done()

            except queue.Empty:
                # No task available, continue
                continue
            except Exception as e:
                logger.error(f"Error in model thread: {str(e)}")
                # Put error in result queue
                self.result_queue.put(
                    {"id": task.get("id", "unknown"), "error": str(e)}
                )
                # Mark task as done
                self.queue.task_done()


class PDFFormAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Form Annotator with LayoutLMv3")
        self.root.geometry("1280x720")

        # Application state
        self.pdf_document = None
        self.current_page = 0
        self.annotations = {}  # Format: {pdf_path: {page_num: [annotations]}}
        self.current_annotation = None
        self.annotation_mode = False
        self.field_type = tk.StringVar(value="text")
        self.field_name = tk.StringVar(value="")
        self.scale_factor = DEFAULT_SCALE
        self.zoom_level = 100  # Percentage

        # Model state
        self.model = None
        self.processor = None
        self.feature_extractor = None
        self.tokenizer = None
        self.id_to_label = None
        self.label_to_id = None
        self.model_task_queue = queue.Queue()
        self.model_result_queue = queue.Queue()
        self.model_thread = None

        # Training state
        self.training_data = []
        self.is_collecting_training_data = False

        # Setup UI components
        self.setup_ui()

        # Initialize model in a separate thread
        self.init_model_thread = threading.Thread(target=self.init_model)
        self.init_model_thread.daemon = True
        self.init_model_thread.start()

        # Setup result queue checking
        self.root.after(100, self.check_model_results)

    def check_model_results(self):
        """Check for results from the model thread."""
        try:
            while not self.model_result_queue.empty():
                result = self.model_result_queue.get_nowait()

                if "error" in result:
                    messagebox.showerror(
                        "Model Error", f"Error in model processing: {result['error']}"
                    )
                else:
                    # Process result based on ID
                    if result["id"].startswith("test_"):
                        self.process_test_result(result)
                    elif result["id"].startswith("train_"):
                        # Process training result
                        pass

                self.model_result_queue.task_done()
        except Exception as e:
            logger.error(f"Error checking model results: {str(e)}")

        # Schedule next check
        self.root.after(100, self.check_model_results)

    def setup_ui(self):
        # Main menu
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # File menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open PDF", command=self.open_pdf)
        self.file_menu.add_command(
            label="Save Annotations", command=self.save_annotations
        )
        self.file_menu.add_command(
            label="Load Annotations", command=self.load_annotations
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)

        # Model menu
        self.model_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Model", menu=self.model_menu)
        self.model_menu.add_command(label="Train Model", command=self.train_model)
        self.model_menu.add_command(label="Load Model", command=self.load_custom_model)
        self.model_menu.add_command(label="Save Model", command=self.save_model)
        self.model_menu.add_separator()
        self.model_menu.add_command(
            label="Collect Training Data", command=self.toggle_collect_training
        )

        # Help menu
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="About", command=self.show_about)

        # Main layout - split into left panel (PDF view) and right panel (controls)
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - PDF viewer
        self.pdf_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(self.pdf_frame, weight=3)

        # PDF canvas with scrollbars
        self.canvas_frame = ttk.Frame(self.pdf_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set,
            bg="gray80",
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)

        # Canvas event bindings for annotations
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows and macOS
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux scroll down

        # PDF navigation controls
        self.nav_frame = ttk.Frame(self.pdf_frame)
        self.nav_frame.pack(fill=tk.X, padx=5, pady=5)

        self.prev_btn = ttk.Button(
            self.nav_frame, text="Previous Page", command=self.prev_page
        )
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.page_label = ttk.Label(self.nav_frame, text="Page: 0/0")
        self.page_label.pack(side=tk.LEFT, padx=5)

        self.next_btn = ttk.Button(
            self.nav_frame, text="Next Page", command=self.next_page
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)

        # Zoom controls
        self.zoom_out_btn = ttk.Button(
            self.nav_frame, text="Zoom Out", command=self.zoom_out
        )
        self.zoom_out_btn.pack(side=tk.RIGHT, padx=5)

        self.zoom_label = ttk.Label(self.nav_frame, text="100%")
        self.zoom_label.pack(side=tk.RIGHT, padx=5)

        self.zoom_in_btn = ttk.Button(
            self.nav_frame, text="Zoom In", command=self.zoom_in
        )
        self.zoom_in_btn.pack(side=tk.RIGHT, padx=5)

        # Right panel - Controls
        self.control_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(self.control_frame, weight=1)

        # Status bar
        self.status_bar = ttk.Label(
            self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Notebook for different control panels
        self.notebook = ttk.Notebook(self.control_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Annotation tab
        self.annotation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.annotation_tab, text="Annotation")

        # Field type selector
        ttk.Label(self.annotation_tab, text="Field Type:").pack(
            anchor=tk.W, padx=5, pady=2
        )
        self.field_type_combo = ttk.Combobox(
            self.annotation_tab,
            textvariable=self.field_type,
            values=FIELD_TYPES,
            state="readonly",
        )
        self.field_type_combo.pack(fill=tk.X, padx=5, pady=2)

        # Field name entry
        ttk.Label(self.annotation_tab, text="Field Name:").pack(
            anchor=tk.W, padx=5, pady=2
        )
        self.field_name_entry = ttk.Entry(
            self.annotation_tab, textvariable=self.field_name
        )
        self.field_name_entry.pack(fill=tk.X, padx=5, pady=2)

        # Annotation mode toggle
        self.annotate_btn = ttk.Button(
            self.annotation_tab,
            text="Start Annotation",
            command=self.toggle_annotation_mode,
        )
        self.annotate_btn.pack(fill=tk.X, padx=5, pady=5)

        # Delete selected annotation
        self.delete_btn = ttk.Button(
            self.annotation_tab,
            text="Delete Selected",
            command=self.delete_selected_annotation,
        )
        self.delete_btn.pack(fill=tk.X, padx=5, pady=5)

        # Annotations list
        ttk.Label(self.annotation_tab, text="Annotations:").pack(
            anchor=tk.W, padx=5, pady=2
        )

        self.annotations_frame = ttk.Frame(self.annotation_tab)
        self.annotations_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.annotations_tree = ttk.Treeview(
            self.annotations_frame, columns=("name", "type", "page"), show="headings"
        )
        self.annotations_tree.heading("name", text="Field Name")
        self.annotations_tree.heading("type", text="Field Type")
        self.annotations_tree.heading("page", text="Page")
        self.annotations_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self.annotations_scrollbar = ttk.Scrollbar(
            self.annotations_frame,
            orient="vertical",
            command=self.annotations_tree.yview,
        )
        self.annotations_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.annotations_tree.configure(yscrollcommand=self.annotations_scrollbar.set)

        # Bind click on annotation list
        self.annotations_tree.bind("<<TreeviewSelect>>", self.on_annotation_select)

        # Test tab
        self.test_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.test_tab, text="Testing")

        # Test controls
        ttk.Button(
            self.test_tab, text="Test PDF Form", command=self.test_pdf_form
        ).pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(self.test_tab, text="Batch Test", command=self.batch_test).pack(
            fill=tk.X, padx=5, pady=5
        )

        # Test results section
        ttk.Label(self.test_tab, text="Test Results:").pack(anchor=tk.W, padx=5, pady=2)

        self.results_frame = ttk.Frame(self.test_tab)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.results_text = tk.Text(self.results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self.results_scrollbar = ttk.Scrollbar(
            self.results_frame, orient="vertical", command=self.results_text.yview
        )
        self.results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.configure(yscrollcommand=self.results_scrollbar.set)

        # Training tab
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="Training")

        # Training controls
        ttk.Label(self.training_tab, text="Model Training:").pack(
            anchor=tk.W, padx=5, pady=2
        )

        self.train_btn = ttk.Button(
            self.training_tab, text="Start Training", command=self.train_model
        )
        self.train_btn.pack(fill=tk.X, padx=5, pady=5)

        self.collect_data_btn = ttk.Button(
            self.training_tab,
            text="Collect Training Data",
            command=self.toggle_collect_training,
        )
        self.collect_data_btn.pack(fill=tk.X, padx=5, pady=5)

        # Training progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.training_tab, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

        self.train_status = ttk.Label(self.training_tab, text="Not training")
        self.train_status.pack(padx=5, pady=5)

        # Training log
        ttk.Label(self.training_tab, text="Training Log:").pack(
            anchor=tk.W, padx=5, pady=2
        )

        self.train_log_frame = ttk.Frame(self.training_tab)
        self.train_log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.train_log = tk.Text(self.train_log_frame, wrap=tk.WORD)
        self.train_log.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self.train_log_scrollbar = ttk.Scrollbar(
            self.train_log_frame, orient="vertical", command=self.train_log.yview
        )
        self.train_log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.train_log.configure(yscrollcommand=self.train_log_scrollbar.set)


def init_model(self):
    """Initialize the LayoutLMv3 model with built-in OCR."""
    try:
        # Update status
        self.update_status("Loading LayoutLMv3 model...")

        # Check if we have a fine-tuned model
        if os.path.exists(MODEL_PATH):
            # Load fine-tuned model and processor WITH apply_ocr=True
            self.processor = LayoutLMv3Processor.from_pretrained(
                MODEL_PATH, apply_ocr=True  # Use built-in OCR
            )
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)

            # Load id_to_label mapping
            if os.path.exists(os.path.join(MODEL_PATH, "id_to_label.json")):
                with open(os.path.join(MODEL_PATH, "id_to_label.json"), "r") as f:
                    self.id_to_label = json.load(f)
                    self.label_to_id = {v: k for k, v in self.id_to_label.items()}

            logger.info("Loaded fine-tuned model with OCR")
            self.update_status("Fine-tuned model with OCR loaded")
        else:
            # Initialize from pretrained WITH apply_ocr=True
            self.processor = LayoutLMv3Processor.from_pretrained(
                "microsoft/layoutlmv3-base", apply_ocr=True  # Use built-in OCR
            )

            # For token classification (form field extraction)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                "microsoft/layoutlmv3-base"
            )

            # Create label mapping
            self.id_to_label = {0: "O"}  # Outside
            for i, field_type in enumerate(FIELD_TYPES):
                self.id_to_label[i + 1] = f"B-{field_type}"  # Beginning of field
                self.id_to_label[i + 1 + len(FIELD_TYPES)] = (
                    f"I-{field_type}"  # Inside field
                )

            self.label_to_id = {v: k for k, v in self.id_to_label.items()}

            logger.info("Loaded base model with OCR")
            self.update_status("Base model with OCR loaded")

        # Start model thread
        self.model_thread = ModelThread(
            self.model_task_queue,
            self.model_result_queue,
            self.processor,
            self.model,
        )
        self.model_thread.start()

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        self.update_status(f"Error loading model: {str(e)}")
        messagebox.showerror("Model Error", f"Error loading LayoutLMv3 model: {str(e)}")

    def update_status(self, message):
        """Update status bar with a message."""

        def _update():
            self.status_bar.config(text=message)

        # Update from main thread
        if threading.current_thread() is threading.main_thread():
            _update()
        else:
            self.root.after(0, _update)

    def open_pdf(self):
        """Open a PDF file for annotation."""
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            # Close current document if any
            if self.pdf_document:
                self.pdf_document.close()

            self.pdf_document = fitz.open(file_path)
            self.current_page = 0

            # Initialize annotations for this document if not already present
            if file_path not in self.annotations:
                self.annotations[file_path] = {}

            # Refresh view
            self.load_current_page()

            # Update status
            filename = os.path.basename(file_path)
            self.update_status(
                f"Opened {filename} ({self.pdf_document.page_count} pages)"
            )

            # Clear and update annotation list
            self.refresh_annotation_list()

        except Exception as e:
            logger.error(f"Error opening PDF: {str(e)}")
            messagebox.showerror("Error", f"Error opening PDF: {str(e)}")

    def load_current_page(self):
        """Load and display the current page of the PDF."""
        if not self.pdf_document:
            return

        # Update page label
        self.page_label.config(
            text=f"Page: {self.current_page + 1}/{self.pdf_document.page_count}"
        )

        # Get current page
        page = self.pdf_document[self.current_page]

        # Calculate scale factor based on zoom level
        scale = self.scale_factor * (self.zoom_level / 100)

        # Render page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))

        # Convert to PIL Image
        img_data = pix.samples
        img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
        self.photo = ImageTk.PhotoImage(image=img)

        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, img.width, img.height))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Draw existing annotations for this page
        self.draw_annotations()

        # Store page image for possible OCR
        self.current_page_image = img

    def draw_annotations(self):
        """Draw all annotations for the current page on the canvas."""
        if not self.pdf_document:
            return

        # Clear existing annotation rectangles
        self.canvas.delete("annotation")

        # Draw all annotations for the current page
        pdf_path = self.pdf_document.name
        if (
            pdf_path in self.annotations
            and self.current_page in self.annotations[pdf_path]
        ):
            for i, annotation in enumerate(
                self.annotations[pdf_path][self.current_page]
            ):
                # Get display coordinates
                display = annotation.display

                # Color based on field type
                color = self.get_field_color(annotation.field_type)

                # Draw rectangle
                rect_id = self.canvas.create_rectangle(
                    display["x1"],
                    display["y1"],
                    display["x2"],
                    display["y2"],
                    outline=color,
                    width=2,
                    tags=("annotation", f"annotation_{i}"),
                )

                # Draw label
                text_id = self.canvas.create_text(
                    display["x1"],
                    display["y1"] - 10,
                    text=f"{annotation.name} ({annotation.field_type})",
                    anchor=tk.W,
                    fill=color,
                    tags=("annotation", f"annotation_{i}"),
                )

    def get_field_color(self, field_type):
        """Return a color based on field type."""
        colors = {
            "text": "blue",
            "checkbox": "green",
            "radio": "purple",
            "dropdown": "orange",
            "signature": "red",
            "date": "brown",
            "number": "navy",
            "currency": "darkgreen",
        }
        return colors.get(field_type, "gray")

    def next_page(self):
        """Go to the next page of the PDF."""
        if self.pdf_document and self.current_page < self.pdf_document.page_count - 1:
            self.current_page += 1
            self.load_current_page()

    def prev_page(self):
        """Go to the previous page of the PDF."""
        if self.pdf_document and self.current_page > 0:
            self.current_page -= 1
            self.load_current_page()

    def zoom_in(self):
        """Zoom in the PDF view."""
        if self.zoom_level < 300:  # Max 300%
            self.zoom_level += 25
            self.zoom_label.config(text=f"{self.zoom_level}%")
            self.load_current_page()

    def zoom_out(self):
        """Zoom out the PDF view."""
        if self.zoom_level > 25:  # Min 25%
            self.zoom_level -= 25
            self.zoom_label.config(text=f"{self.zoom_level}%")
            self.load_current_page()

    def on_mouse_wheel(self, event):
        """Handle mouse wheel events for zooming."""
        # Determine direction based on event
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            # Scroll up - zoom in
            self.zoom_in()
        elif event.num == 5 or (hasattr(event, "delta") and event.delta < 0):
            # Scroll down - zoom out
            self.zoom_out()

    def toggle_annotation_mode(self):
        """Toggle annotation mode on/off."""
        self.annotation_mode = not self.annotation_mode

        if self.annotation_mode:
            self.annotate_btn.config(text="Stop Annotation")
            self.canvas.config(cursor="crosshair")
            self.update_status(
                "Annotation mode active - draw rectangles around form fields"
            )
        else:
            self.annotate_btn.config(text="Start Annotation")
            self.canvas.config(cursor="")
            self.update_status("Annotation mode disabled")

    def toggle_collect_training(self):
        """Toggle training data collection mode."""
        self.is_collecting_training_data = not self.is_collecting_training_data

        if self.is_collecting_training_data:
            self.collect_data_btn.config(text="Stop Collecting")
            self.update_status(
                "Collecting training data - annotate fields and their values"
            )

            # Display dialog for entering field values
            if self.pdf_document:
                self.show_value_entry_dialog()
        else:
            self.collect_data_btn.config(text="Collect Training Data")
            self.update_status("Training data collection stopped")

    def show_value_entry_dialog(self):
        """Show dialog for entering field values during training data collection."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Enter Field Values")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Enter values for annotated fields:").pack(
            padx=10, pady=10
        )

        # Create a scrollable frame
        canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")

        # Create entries for each field
        entries = {}
        pdf_path = self.pdf_document.name

        if pdf_path in self.annotations:
            for page_num, page_annotations in self.annotations[pdf_path].items():
                for annotation in page_annotations:
                    frame = ttk.Frame(scrollable_frame)
                    frame.pack(fill=tk.X, padx=5, pady=2)

                    # Field label with page number
                    label_text = f"{annotation.name} (Page {page_num+1}, {annotation.field_type})"
                    ttk.Label(frame, text=label_text).pack(side=tk.LEFT, padx=5)

                    # Entry for field value
                    entry = ttk.Entry(frame)
                    entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

                    # Pre-fill with existing label if any
                    if annotation.label:
                        entry.insert(0, annotation.label)

                    entries[(page_num, annotation.name)] = entry

        # Buttons frame
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            btn_frame,
            text="Save Values",
            command=lambda: self.save_training_field_values(entries, dialog),
        ).pack(side=tk.RIGHT, padx=5)

        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.RIGHT, padx=5
        )

    def save_training_field_values(self, entries, dialog):
        """Save field values for training data."""
        pdf_path = self.pdf_document.name

        for (page_num, field_name), entry in entries.items():
            # Find the annotation
            if page_num in self.annotations[pdf_path]:
                for annotation in self.annotations[pdf_path][page_num]:
                    if annotation.name == field_name:
                        # Update the label
                        annotation.label = entry.get()

        # Close dialog
        dialog.destroy()

        # Update status
        self.update_status("Field values saved for training")

        # Refresh annotation list
        self.refresh_annotation_list()

    def on_canvas_click(self, event):
        """Handle mouse click on canvas - start annotation or select existing."""
        if not self.pdf_document:
            return

        # Get canvas coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # Check if clicking on an existing annotation
        items = self.canvas.find_overlapping(x - 5, y - 5, x + 5, y + 5)
        for item in items:
            tags = self.canvas.gettags(item)
            for tag in tags:
                if tag.startswith("annotation_"):
                    # Get annotation index
                    idx = int(tag.split("_")[1])

                    # Select this annotation
                    pdf_path = self.pdf_document.name
                    if (
                        pdf_path in self.annotations
                        and self.current_page in self.annotations[pdf_path]
                    ):
                        if idx < len(self.annotations[pdf_path][self.current_page]):
                            # Highlight the annotation on canvas
                            self.canvas.delete("selection")

                            annotation = self.annotations[pdf_path][self.current_page][
                                idx
                            ]
                            display = annotation.display

                            self.canvas.create_rectangle(
                                display["x1"],
                                display["y1"],
                                display["x2"],
                                display["y2"],
                                outline="yellow",
                                width=3,
                                tags="selection",
                            )

                            # Select in tree view
                            self.select_annotation_in_tree(annotation)

                            return

        # If in annotation mode, start a new annotation
        if self.annotation_mode:
            # Start a new annotation
            self.current_annotation = {
                "start_x": x,
                "start_y": y,
                "end_x": x,
                "end_y": y,
            }

            # Draw initial rectangle
            self.canvas.delete("new_annotation")
            self.current_rect = self.canvas.create_rectangle(
                x, y, x, y, outline="red", width=2, tags="new_annotation"
            )

    def on_canvas_drag(self, event):
        """Handle mouse drag on canvas - update annotation rectangle."""
        if not self.annotation_mode or not self.current_annotation:
            return

        # Update rectangle end coordinates
        self.current_annotation["end_x"] = self.canvas.canvasx(event.x)
        self.current_annotation["end_y"] = self.canvas.canvasy(event.y)

        # Update rectangle display
        self.canvas.coords(
            self.current_rect,
            self.current_annotation["start_x"],
            self.current_annotation["start_y"],
            self.current_annotation["end_x"],
            self.current_annotation["end_y"],
        )

    def on_canvas_release(self, event):
        """Handle mouse release - finalize annotation."""
        if not self.annotation_mode or not self.current_annotation:
            return

        # Finalize the annotation coordinates
        x1 = min(self.current_annotation["start_x"], self.current_annotation["end_x"])
        y1 = min(self.current_annotation["start_y"], self.current_annotation["end_y"])
        x2 = max(self.current_annotation["start_x"], self.current_annotation["end_x"])
        y2 = max(self.current_annotation["start_y"], self.current_annotation["end_y"])

        # Check if the rectangle is too small (likely a click rather than drag)
        if (x2 - x1 < 10) or (y2 - y1 < 10):
            self.canvas.delete("new_annotation")
            self.current_annotation = None
            return

        # Create normalized box coordinates (0-1000 scale for LayoutLM)
        page = self.pdf_document[self.current_page]
        width, height = page.rect.width, page.rect.height
        scale = self.scale_factor * (self.zoom_level / 100)  # Account for zoom level

        # Normalize to 0-1000 range for LayoutLM
        bbox = [
            int(x1 / (width * scale) * 1000),
            int(y1 / (height * scale) * 1000),
            int(x2 / (width * scale) * 1000),
            int(y2 / (height * scale) * 1000),
        ]

        # Validate field name
        field_name = self.field_name.get().strip()
        if not field_name:
            messagebox.showwarning("Missing Information", "Please enter a field name.")
            self.canvas.delete("new_annotation")
            self.current_annotation = None
            return

        # Store annotation data
        pdf_path = self.pdf_document.name

        if self.current_page not in self.annotations[pdf_path]:
            self.annotations[pdf_path][self.current_page] = []

        # Check for field name uniqueness
        for page_num, page_annotations in self.annotations[pdf_path].items():
            for existing_annotation in page_annotations:
                if existing_annotation.name == field_name:
                    messagebox.showwarning(
                        "Duplicate Field",
                        f"Field '{field_name}' already exists on page {page_num+1}. Please use a unique name.",
                    )
                    self.canvas.delete("new_annotation")
                    self.current_annotation = None
                    return

        # Create annotation object
        annotation = FormFieldAnnotation(
            name=field_name,
            field_type=self.field_type.get(),
            bbox=bbox,
            page=self.current_page,
            display={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        )

        self.annotations[pdf_path][self.current_page].append(annotation)

        # Add to tree view
        self.annotations_tree.insert(
            "",
            tk.END,
            values=(annotation.name, annotation.field_type, self.current_page + 1),
        )

        # Reset current annotation
        self.current_annotation = None

        # Redraw all annotations
        self.draw_annotations()

        # If collecting training data, prompt for field value
        if self.is_collecting_training_data:
            self.prompt_for_field_value(annotation)

    def save_annotations(self):
        """Save annotations to a JSON file."""
        if not self.pdf_document:
            messagebox.showwarning("Warning", "No PDF document open")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"{os.path.splitext(os.path.basename(self.pdf_document.name))[0]}_annotations.json",
        )

        if not file_path:
            return

        try:
            # Convert to serializable format
            serializable_data = {}
            for pdf_path, pages in self.annotations.items():
                serializable_data[pdf_path] = {}
                for page_num, annotations in pages.items():
                    serializable_data[pdf_path][str(page_num)] = [
                        {
                            "name": ann.name,
                            "field_type": ann.field_type,
                            "bbox": ann.bbox,
                            "page": ann.page,
                            "display": ann.display,
                            "label": ann.label,
                        }
                        for ann in annotations
                    ]

            with open(file_path, "w") as f:
                json.dump(serializable_data, f, indent=4)

            self.update_status(f"Annotations saved to {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Error saving annotations: {str(e)}")
            messagebox.showerror("Error", f"Error saving annotations: {str(e)}")

    def load_annotations(self):
        """Load annotations from a JSON file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Convert to annotation objects
            loaded_annotations = {}
            for pdf_path, pages in data.items():
                loaded_annotations[pdf_path] = {}
                for page_num, annotations in pages.items():
                    loaded_annotations[pdf_path][int(page_num)] = [
                        FormFieldAnnotation(
                            name=ann["name"],
                            field_type=ann["field_type"],
                            bbox=ann["bbox"],
                            page=ann["page"],
                            display=ann["display"],
                            label=ann.get("label"),
                        )
                        for ann in annotations
                    ]

            # Merge with existing annotations
            if messagebox.askyesno(
                "Merge Annotations",
                "Merge with existing annotations? Select 'No' to replace all annotations.",
            ):
                # Merge
                for pdf_path, pages in loaded_annotations.items():
                    if pdf_path not in self.annotations:
                        self.annotations[pdf_path] = {}

                    for page_num, annotations in pages.items():
                        if page_num not in self.annotations[pdf_path]:
                            self.annotations[pdf_path][page_num] = []

                        # Add annotations, avoiding duplicates by name
                        existing_names = {
                            a.name for a in self.annotations[pdf_path][page_num]
                        }
                        for annotation in annotations:
                            if annotation.name not in existing_names:
                                self.annotations[pdf_path][page_num].append(annotation)
                                existing_names.add(annotation.name)
            else:
                # Replace
                self.annotations = loaded_annotations

            # Refresh annotations list
            self.refresh_annotation_list()

            # Redraw current page
            self.draw_annotations()

            self.update_status(f"Annotations loaded from {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Error loading annotations: {str(e)}")
            messagebox.showerror("Error", f"Error loading annotations: {str(e)}")

    def refresh_annotation_list(self):
        """Refresh the annotation list tree view."""
        # Clear existing items
        self.annotations_tree.delete(*self.annotations_tree.get_children())

        # Add all annotations
        if not self.pdf_document:
            return

        pdf_path = self.pdf_document.name
        if pdf_path in self.annotations:
            for page_num, page_annotations in self.annotations[pdf_path].items():
                for annotation in page_annotations:
                    # Format label/value if present
                    label_info = f" = {annotation.label}" if annotation.label else ""

                    self.annotations_tree.insert(
                        "",
                        tk.END,
                        values=(annotation.name, annotation.field_type, page_num + 1),
                    )

    def select_annotation_in_tree(self, annotation):
        """Select an annotation in the treeview."""
        for item in self.annotations_tree.get_children():
            values = self.annotations_tree.item(item, "values")
            if values[0] == annotation.name and int(values[2]) == self.current_page + 1:
                self.annotations_tree.selection_set(item)
                self.annotations_tree.see(item)
                return

    def load_custom_model(self):
        """Load a custom trained model."""
        model_dir = filedialog.askdirectory(title="Select Model Directory")
        if not model_dir:
            return

        try:
            # Load model and processor
            self.processor = LayoutLMv3Processor.from_pretrained(model_dir)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)

            # Load id_to_label mapping
            id_to_label_path = os.path.join(model_dir, "id_to_label.json")
            if os.path.exists(id_to_label_path):
                with open(id_to_label_path, "r") as f:
                    self.id_to_label = json.load(f)
                    self.label_to_id = {v: k for k, v in self.id_to_label.items()}

            self.update_status(
                f"Loaded custom model from {os.path.basename(model_dir)}"
            )
            messagebox.showinfo("Model Loaded", "Custom model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading custom model: {str(e)}")
            messagebox.showerror("Model Error", f"Error loading custom model: {str(e)}")

    def save_model(self):
        """Save the current model to a directory."""
        if not self.model or not self.processor:
            messagebox.showwarning("Warning", "No model loaded")
            return

        model_dir = filedialog.askdirectory(title="Select Directory to Save Model")
        if not model_dir:
            return

        try:
            # Save model and processor
            self.model.save_pretrained(model_dir)
            self.processor.save_pretrained(model_dir)

            # Save id_to_label mapping
            with open(os.path.join(model_dir, "id_to_label.json"), "w") as f:
                json.dump(self.id_to_label, f, indent=4)

            self.update_status(f"Saved model to {os.path.basename(model_dir)}")
            messagebox.showinfo("Model Saved", "Model saved successfully!")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            messagebox.showerror("Model Error", f"Error saving model: {str(e)}")

    def delete_selected_annotation(self):
        """Delete the selected annotation."""
        if not self.pdf_document:
            return

        # Get selected item
        selection = self.annotations_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select an annotation to delete.")
            return

        item = selection[0]
        values = self.annotations_tree.item(item, "values")

        field_name = values[0]
        page_num = int(values[2]) - 1

        # Confirm deletion
        if not messagebox.askyesno("Confirm", f"Delete annotation '{field_name}'?"):
            return

        # Remove annotation
        pdf_path = self.pdf_document.name
        if pdf_path in self.annotations and page_num in self.annotations[pdf_path]:
            self.annotations[pdf_path][page_num] = [
                annotation
                for annotation in self.annotations[pdf_path][page_num]
                if annotation.name != field_name
            ]

        # Remove from tree view
        self.annotations_tree.delete(item)

        # Clear selection highlighting
        self.canvas.delete("selection")

        # Redraw annotations if on the current page
        if page_num == self.current_page:
            self.draw_annotations()

    def on_annotation_select(self, event):
        """Handle selection of an annotation in the tree view."""
        if not self.pdf_document:
            return

        # Get selected item
        selection = self.annotations_tree.selection()
        if not selection:
            return

        item = selection[0]
        values = self.annotations_tree.item(item, "values")

        field_name = values[0]
        page_num = int(values[2]) - 1

        # If selected annotation is on a different page, switch to that page
        if page_num != self.current_page:
            self.current_page = page_num
            self.load_current_page()

        # Highlight the annotation on canvas
        pdf_path = self.pdf_document.name
        if pdf_path in self.annotations and page_num in self.annotations[pdf_path]:
            for annotation in self.annotations[pdf_path][page_num]:
                if annotation.name == field_name:
                    # Highlight on canvas
                    self.canvas.delete("selection")

                    display = annotation.display
                    self.canvas.create_rectangle(
                        display["x1"],
                        display["y1"],
                        display["x2"],
                        display["y2"],
                        outline="yellow",
                        width=3,
                        tags="selection",
                    )

                    # Show field properties
                    self.field_name.set(annotation.name)
                    self.field_type.set(annotation.field_type)

                    # Center view on the annotation
                    self.canvas.xview_moveto(
                        (display["x1"] - 50) / self.canvas.winfo_width()
                    )
                    self.canvas.yview_moveto(
                        (display["y1"] - 50) / self.canvas.winfo_height()
                    )

                    break

    def show_about(self):
        """Show about dialog."""
        about_window = tk.Toplevel(self.root)
        about_window.title("About PDF Form Annotator")
        about_window.geometry("400x300")
        about_window.transient(self.root)
        about_window.grab_set()

        ttk.Label(
            about_window, text="PDF Form Annotator", font=("Helvetica", 16, "bold")
        ).pack(padx=10, pady=10)

        ttk.Label(about_window, text="Version 1.0.0").pack(padx=10, pady=5)

        ttk.Label(
            about_window,
            text="A tool for annotating and extracting data from PDF forms\nusing the LayoutLMv3 model.",
            justify=tk.CENTER,
        ).pack(padx=10, pady=5)

        ttk.Label(about_window, text="© 2025", justify=tk.CENTER).pack(padx=10, pady=5)

        ttk.Button(about_window, text="Close", command=about_window.destroy).pack(
            padx=10, pady=20
        )

    def test_pdf_form(self):
        """Test a PDF form with the trained model."""
        if not self.model or not self.processor:
            messagebox.showwarning("Warning", "LayoutLMv3 model not loaded")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            # Open test PDF
            test_doc = fitz.open(file_path)

            # Clear results display
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(
                tk.END, f"Analyzing {os.path.basename(file_path)}...\n\n"
            )

            # Process each page
            results = {}
            for page_num in range(
                min(test_doc.page_count, 10)
            ):  # Limit to first 10 pages
                self.results_text.insert(tk.END, f"Processing page {page_num + 1}...\n")
                self.root.update_idletasks()  # Update UI

                # Get page
                page = test_doc[page_num]

                # Extract text and boxes
                words, boxes = self.extract_text_from_page(page)

                if not words:
                    self.results_text.insert(tk.END, "  No text found on this page.\n")
                    continue

                # Convert page to image
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Queue task for model thread
                task_id = f"test_{page_num}"
                self.model_task_queue.put(
                    {
                        "type": "inference",
                        "id": task_id,
                        "image": img,
                        "words": words,
                        "boxes": boxes,
                    }
                )

                self.results_text.insert(
                    tk.END, f"  Queued page {page_num + 1} for processing...\n"
                )

            messagebox.showinfo(
                "Processing",
                "PDF queued for processing. Results will appear in the Test Results panel.",
            )

        except Exception as e:
            logger.error(f"Error processing test PDF: {str(e)}")
            messagebox.showerror("Error", f"Error processing test PDF: {str(e)}")

    def extract_text_from_page(self, page):
        """Extract text and bounding boxes from a PDF page."""
        # Get text from the page with bounding boxes
        text_blocks = page.get_text("dict")["blocks"]

        words = []
        boxes = []

        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        for span in line["spans"]:
                            # Extract text
                            text = span["text"].strip()
                            if not text:
                                continue

                            # Split into words
                            for word in text.split():
                                if not word:
                                    continue

                                words.append(word)

                                # Get bounding box and normalize to 0-1000 range
                                x0, y0, x1, y1 = span["bbox"]
                                width, height = page.rect.width, page.rect.height

                                boxes.append(
                                    [
                                        int(x0 / width * 1000),
                                        int(y0 / height * 1000),
                                        int(x1 / width * 1000),
                                        int(y1 / height * 1000),
                                    ]
                                )

        return words, boxes

    def process_test_result(self, result):
        """Process a test result from the model thread."""
        predictions = result["predictions"]
        words = result["words"]
        boxes = result["boxes"]
        page_num = int(result["id"].split("_")[1])

        # Group words by field type
        field_values = {}
        current_field = None
        current_text = ""

        for i, (word, pred) in enumerate(zip(words, predictions)):
            label = self.id_to_label.get(pred, "O")

            # Check if this is a beginning of a field
            if label.startswith("B-"):
                # Save the previous field if any
                if current_field and current_text:
                    if current_field not in field_values:
                        field_values[current_field] = []
                    field_values[current_field].append(current_text.strip())

                # Start new field
                current_field = label[2:]  # Remove "B-" prefix
                current_text = word
            # Check if this is inside a field
            elif label.startswith("I-") and current_field == label[2:]:
                current_text += " " + word
            # Check if this is outside any field
            elif label == "O":
                # Save the previous field if any
                if current_field and current_text:
                    if current_field not in field_values:
                        field_values[current_field] = []
                    field_values[current_field].append(current_text.strip())

                # Reset
                current_field = None
                current_text = ""

        # Save the last field if any
        if current_field and current_text:
            if current_field not in field_values:
                field_values[current_field] = []
            field_values[current_field].append(current_text.strip())

        # Display results
        self.results_text.insert(tk.END, f"\nResults for page {page_num + 1}:\n")

        if not field_values:
            self.results_text.insert(tk.END, "  No form fields detected.\n")
        else:
            for field_type, values in field_values.items():
                for value in values:
                    self.results_text.insert(tk.END, f"  {field_type}: {value}\n")

        self.results_text.insert(tk.END, "\n")
        self.results_text.see(tk.END)

    def train_model(self):
        """Train the LayoutLMv3 model on the current annotations."""
        if not self.model or not self.processor:
            messagebox.showwarning("Warning", "LayoutLMv3 model not loaded")
            return

        # Check if we have any training data
        has_training_data = False
        for pdf_path, pages in self.annotations.items():
            for page_num, annotations in pages.items():
                for annotation in annotations:
                    if annotation.label:
                        has_training_data = True
                        break

        if not has_training_data:
            messagebox.showwarning(
                "No Training Data",
                "No field values found. Please annotate field values first using 'Collect Training Data'.",
            )
            return

        # Confirm training
        if not messagebox.askyesno(
            "Start Training", "Training the model may take some time. Continue?"
        ):
            return

        # Show training progress dialog
        training_window = tk.Toplevel(self.root)
        training_window.title("Model Training")
        training_window.geometry("500x400")
        training_window.transient(self.root)
        training_window.grab_set()

        ttk.Label(training_window, text="Training LayoutLMv3 Model").pack(
            padx=10, pady=10
        )

        # Progress bar
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            training_window, variable=progress_var, maximum=100
        )
        progress_bar.pack(fill=tk.X, padx=10, pady=5)

        # Status label
        status_label = ttk.Label(training_window, text="Preparing training data...")
        status_label.pack(padx=10, pady=5)

        # Log text
        log_frame = ttk.Frame(training_window)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        log_text = tk.Text(log_frame, wrap=tk.WORD, height=15)
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        log_scrollbar = ttk.Scrollbar(
            log_frame, orient="vertical", command=log_text.yview
        )
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        log_text.configure(yscrollcommand=log_scrollbar.set)

        # Function to add log message
        def add_log(message):
            log_text.insert(tk.END, message + "\n")
            log_text.see(tk.END)
            log_text.update_idletasks()

        # Function to prepare training data
        def prepare_training_data():
            add_log("Preparing training data...")
            training_data = []

            for pdf_path, pages in self.annotations.items():
                add_log(f"Processing annotations from {os.path.basename(pdf_path)}")

                try:
                    # Open the PDF
                    doc = fitz.open(pdf_path)

                    for page_num, annotations in pages.items():
                        add_log(f"  Processing page {page_num + 1}")

                        # Skip pages without annotated values
                        has_labels = any(a.label for a in annotations)
                        if not has_labels:
                            add_log(
                                f"    No labeled annotations on page {page_num + 1}, skipping"
                            )
                            continue

                        # Get the page
                        page = doc[page_num]

                        # Extract text and boxes
                        words, boxes = self.extract_text_from_page(page)

                        if not words:
                            add_log(
                                f"    No text found on page {page_num + 1}, skipping"
                            )
                            continue

                        # Convert page to image
                        pix = page.get_pixmap()
                        img = Image.frombytes(
                            "RGB", [pix.width, pix.height], pix.samples
                        )

                        # Create labels for each word based on annotations
                        labels = ["O"] * len(words)  # Default to "Outside"

                        for annotation in annotations:
                            if not annotation.label:
                                continue

                            # Find words that match the label
                            label_words = annotation.label.split()

                            for i in range(len(words) - len(label_words) + 1):
                                if words[i : i + len(label_words)] == label_words:
                                    # Mark as field type
                                    labels[i] = (
                                        f"B-{annotation.field_type}"  # Beginning
                                    )

                                    for j in range(1, len(label_words)):
                                        labels[i + j] = (
                                            f"I-{annotation.field_type}"  # Inside
                                        )

                        # Convert labels to IDs
                        label_ids = [self.label_to_id.get(label, 0) for label in labels]

                        # Add to training data
                        training_data.append(
                            {
                                "image": img,
                                "words": words,
                                "boxes": boxes,
                                "labels": label_ids,
                            }
                        )

                        add_log(
                            f"    Added {len(words)} words with {sum(1 for l in labels if l != 'O')} field markers"
                        )

                    doc.close()

                except Exception as e:
                    add_log(f"Error processing {pdf_path}: {str(e)}")
                    logger.error(f"Error processing {pdf_path}: {str(e)}")

            add_log(f"Prepared {len(training_data)} pages for training")
            return training_data

        def prepare_training_data_ocr(self):
            """Prepare training data using OCR approach."""
            add_log = self.add_log if hasattr(self, "add_log") else print
            add_log("Preparing training data with OCR approach...")
            training_data = []

            for pdf_path, pages in self.annotations.items():
                add_log(f"Processing annotations from {os.path.basename(pdf_path)}")

                try:
                    # Open the PDF
                    doc = fitz.open(pdf_path)

                    for page_num, annotations in pages.items():
                        add_log(f"  Processing page {page_num + 1}")

                        # Skip pages without annotated values
                        has_labels = any(a.label for a in annotations)
                        if not has_labels:
                            add_log(
                                f"    No labeled annotations on page {page_num + 1}, skipping"
                            )
                            continue

                        # Get the page
                        page = doc[page_num]

                        # Convert page to image
                        pix = page.get_pixmap()
                        img = Image.frombytes(
                            "RGB", [pix.width, pix.height], pix.samples
                        )

                        # First, process the image with the processor to get OCR results
                        encoding = self.processor(images=img, return_tensors="pt")

                        # Extract OCR words and their boxes
                        ocr_words = self.processor.tokenizer.convert_ids_to_tokens(
                            encoding["input_ids"][0], skip_special_tokens=True
                        )
                        ocr_boxes = encoding["bbox"][0].tolist()

                        # Now, we need to align our annotations with the OCR results
                        # This is a complex task and requires matching annotation labels to OCR detected words

                        # For each annotation with a label, try to find matching words in OCR results
                        labels = ["O"] * len(ocr_words)  # Default to "Outside"

                        for annotation in annotations:
                            if not annotation.label:
                                continue

                            # Simple approach: look for exact matches of the annotation label
                            # in consecutive OCR words (this is a simplification)
                            label_words = annotation.label.lower().split()

                            # Try to find this sequence in OCR words
                            for i in range(len(ocr_words) - len(label_words) + 1):
                                # Check if the sequence matches
                                match = True
                                for j, label_word in enumerate(label_words):
                                    if (
                                        i + j >= len(ocr_words)
                                        or label_word.lower()
                                        not in ocr_words[i + j].lower()
                                    ):
                                        match = False
                                        break

                                if match:
                                    # Found a match! Mark as this field type
                                    labels[i] = (
                                        f"B-{annotation.field_type}"  # Beginning
                                    )
                                    for j in range(1, len(label_words)):
                                        if i + j < len(labels):
                                            labels[i + j] = (
                                                f"I-{annotation.field_type}"  # Inside
                                            )

                        # Convert labels to IDs
                        label_ids = [self.label_to_id.get(label, 0) for label in labels]

                        # Add to training data
                        training_data.append(
                            {
                                "image": img,
                                "ocr_words": ocr_words,
                                "ocr_boxes": ocr_boxes,
                                "labels": label_ids,
                            }
                        )

                        add_log(
                            f"    Added {len(ocr_words)} OCR words with {sum(1 for l in labels if l != 'O')} field markers"
                        )

                    doc.close()

                except Exception as e:
                    add_log(f"Error processing {pdf_path}: {str(e)}")
                    logger.error(f"Error processing {pdf_path}: {str(e)}")

            add_log(f"Prepared {len(training_data)} pages for training")
            return training_data

        # Function to train the model
        def train_model_thread():
            try:
                # Prepare training data
                train_data = prepare_training_data()

                if not train_data:
                    add_log("No training data available. Aborting.")
                    messagebox.showwarning(
                        "Training Error", "No valid training data available."
                    )
                    training_window.destroy()
                    return

                # Create dataset
                dataset = LayoutLMDataset(train_data, self.processor)

                # Training parameters
                num_epochs = 3
                batch_size = 2
                learning_rate = 5e-5

                # Create dataloader
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                # Setup optimizer
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

                # Training loop
                self.model.train()
                for epoch in range(num_epochs):
                    add_log(f"Starting epoch {epoch + 1}/{num_epochs}")
                    epoch_loss = 0

                    for i, batch in enumerate(dataloader):
                        # Update progress
                        progress = (
                            (epoch * len(dataloader) + i)
                            / (num_epochs * len(dataloader))
                            * 100
                        )
                        progress_var.set(progress)
                        status_label.config(
                            text=f"Training: Epoch {epoch + 1}/{num_epochs}, Batch {i+1}/{len(dataloader)}"
                        )
                        training_window.update_idletasks()

                        # Prepare inputs
                        inputs = {k: v for k, v in batch.items() if k != "labels"}
                        labels = batch.get("labels")

                        # Forward pass
                        outputs = self.model(**inputs, labels=labels)
                        loss = outputs.loss

                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()

                        if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                            add_log(
                                f"  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}"
                            )

                    add_log(
                        f"Epoch {epoch + 1}/{num_epochs} complete, Average Loss: {epoch_loss / len(dataloader):.4f}"
                    )

                # Save the model
                os.makedirs(MODEL_PATH, exist_ok=True)
                self.model.save_pretrained(MODEL_PATH)
                self.processor.save_pretrained(MODEL_PATH)

                # Save id_to_label mapping
                with open(os.path.join(MODEL_PATH, "id_to_label.json"), "w") as f:
                    json.dump(self.id_to_label, f, indent=4)

                add_log("Training complete! Model saved.")
                progress_var.set(100)
                status_label.config(text="Training complete!")

                messagebox.showinfo("Training Complete", "Model training complete!")

            except Exception as e:
                add_log(f"Error during training: {str(e)}")
                logger.error(f"Error during training: {str(e)}")
                messagebox.showerror(
                    "Training Error", f"Error during training: {str(e)}"
                )

            finally:
                # Close the training window
                training_window.destroy()

        # Start training in a separate thread
        thread = threading.Thread(target=train_model_thread)
        thread.daemon = True
        thread.start()

        # This completes the truncated batch_test method

    def batch_test(self):
        """Test multiple PDF forms in batch mode."""
        if not self.model or not self.processor:
            messagebox.showwarning("Warning", "LayoutLMv3 model not loaded")
            return

        # Ask for directory containing PDFs
        dir_path = filedialog.askdirectory(title="Select Directory with PDF Forms")
        if not dir_path:
            return

        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return

        # Find all PDF files
        pdf_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))

        if not pdf_files:
            messagebox.showinfo("Info", "No PDF files found in the selected directory.")
            return

        # Show progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Batch Processing")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()

        ttk.Label(progress_window, text="Processing PDF files in batch mode...").pack(
            padx=10, pady=10
        )

        # Progress bar
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            progress_window, variable=progress_var, maximum=len(pdf_files)
        )
        progress_bar.pack(fill=tk.X, padx=10, pady=5)

        # Status label
        status_label = ttk.Label(progress_window, text="Starting...")
        status_label.pack(padx=10, pady=5)

        # Function to process PDFs in batch
        def process_batch():
            results = {}

            for i, pdf_file in enumerate(pdf_files):
                # Update progress
                filename = os.path.basename(pdf_file)
                progress_var.set(i)
                status_label.config(text=f"Processing {filename}...")
                progress_window.update_idletasks()

                try:
                    # Process PDF
                    pdf_results = self.process_pdf_for_batch(pdf_file)
                    results[pdf_file] = pdf_results

                    # Write individual results file
                    output_file = os.path.join(
                        output_dir, f"{os.path.splitext(filename)[0]}_results.json"
                    )

                    with open(output_file, "w") as f:
                        json.dump(pdf_results, f, indent=4)

                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {str(e)}")
                    results[pdf_file] = {"error": str(e)}

            # Write summary file
            summary_file = os.path.join(output_dir, "batch_summary.json")
            with open(summary_file, "w") as f:
                summary = {
                    "total_files": len(pdf_files),
                    "processed_at": datetime.now().isoformat(),
                    "results": {os.path.basename(k): v for k, v in results.items()},
                }
                json.dump(summary, f, indent=4)

            # Close progress window
            progress_window.destroy()

            # Show completion message
            messagebox.showinfo(
                "Batch Complete",
                f"Processed {len(pdf_files)} PDF files. Results saved to {output_dir}",
            )

        # Start processing in a separate thread
        thread = threading.Thread(target=process_batch)
        thread.daemon = True
        thread.start()

    # Add the missing process_pdf_for_batch method
    def process_pdf_for_batch(self, pdf_file):
        """Process a PDF file for batch testing."""
        results = {"pages": {}}

        # Open PDF
        doc = fitz.open(pdf_file)

        # Process each page
        for page_num in range(min(doc.page_count, 10)):  # Limit to first 10 pages
            page = doc[page_num]

            # Extract text and boxes
            words, boxes = self.extract_text_from_page(page)

            if not words:
                results["pages"][str(page_num)] = {"error": "No text found"}
                continue

            # Convert page to image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Process with LayoutLMv3
            encoding = self.processor(
                img,
                words,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self.model(**encoding)
                predictions = outputs.logits.argmax(-1).squeeze().tolist()

            # Extract field values
            field_values = {}
            current_field = None
            current_text = ""

            for i, (word, pred) in enumerate(zip(words, predictions)):
                label = self.id_to_label.get(pred, "O")

                # Check if this is a beginning of a field
                if label.startswith("B-"):
                    # Save the previous field if any
                    if current_field and current_text:
                        if current_field not in field_values:
                            field_values[current_field] = []
                        field_values[current_field].append(current_text.strip())

                    # Start new field
                    current_field = label[2:]  # Remove "B-" prefix
                    current_text = word
                # Check if this is inside a field
                elif label.startswith("I-") and current_field == label[2:]:
                    current_text += " " + word
                # Check if this is outside any field
                elif label == "O":
                    # Save the previous field if any
                    if current_field and current_text:
                        if current_field not in field_values:
                            field_values[current_field] = []
                        field_values[current_field].append(current_text.strip())

                    # Reset
                    current_field = None
                    current_text = ""

            # Save the last field if any
            if current_field and current_text:
                if current_field not in field_values:
                    field_values[current_field] = []
                field_values[current_field].append(current_text.strip())

            # Store results for this page
            results["pages"][str(page_num)] = field_values

        doc.close()
        return results


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = PDFFormAnnotator(root)
    root.mainloop()
