"""
PDF Information Extractor - Annotation Tab
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import fitz  # PyMuPDF
import cv2
import pytesseract
import numpy as np

from pdf_info_extractor.utils.pdf_utils import save_annotations, load_annotations


class AnnotationTab:
    def __init__(self, parent, app):
        """Initialize the annotation tab"""
        self.parent = parent
        self.app = app

        # Create the main frame
        self.frame = ttk.Frame(parent)

        # Application state
        self.current_pdf_path = None
        self.current_page_num = 0
        self.total_pages = 0
        self.current_image = None
        self.current_boxes = []
        self.selected_box_idx = -1
        self.pdf_data = {}  # {pdf_path: {page_num: [boxes]}}
        self.labels = ["O"]  # Initialize with Outside label
        self.label2id = {"O": 0}
        self.id2label = {0: "O"}

        # Set up UI components
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components"""
        # Split into left panel (controls) and right panel (viewer)
        main_frame = ttk.Frame(self.frame)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel
        left_frame = ttk.LabelFrame(main_frame, text="Controls")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # PDF controls
        pdf_frame = ttk.LabelFrame(left_frame, text="PDF Management")
        pdf_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(pdf_frame, text="Open PDF", command=self.open_pdf).pack(
            fill=tk.X, padx=5, pady=2
        )

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

        # OCR controls
        ocr_frame = ttk.LabelFrame(left_frame, text="OCR")
        ocr_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(ocr_frame, text="Run OCR", command=self.run_ocr).pack(
            fill=tk.X, padx=5, pady=2
        )

        # Annotation controls
        annot_frame = ttk.LabelFrame(left_frame, text="Annotation")
        annot_frame.pack(fill=tk.X, padx=5, pady=5)

        label_frame = ttk.Frame(annot_frame)
        label_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(label_frame, text="Label:").pack(side=tk.LEFT, padx=2)

        self.label_var = tk.StringVar()
        self.label_combo = ttk.Combobox(
            label_frame, textvariable=self.label_var, values=self.labels
        )
        self.label_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        ttk.Button(label_frame, text="+", width=3, command=self.add_label).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Button(annot_frame, text="Apply Label", command=self.apply_label).pack(
            fill=tk.X, padx=5, pady=2
        )
        ttk.Button(annot_frame, text="Delete Box", command=self.delete_box).pack(
            fill=tk.X, padx=5, pady=2
        )

        # Save/Load controls
        save_frame = ttk.LabelFrame(left_frame, text="Save/Load")
        save_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            save_frame, text="Save Annotations", command=self.save_annotations
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            save_frame, text="Load Annotations", command=self.load_annotations
        ).pack(fill=tk.X, padx=5, pady=2)

        # Right panel (document view)
        right_frame = ttk.LabelFrame(main_frame, text="Document")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas for PDF viewing
        self.canvas_frame = ttk.Frame(right_frame)
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

        # Canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def open_pdf(self):
        """Open a PDF file for annotation"""
        file_path = filedialog.askopenfilename(
            title="Select PDF file",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )

        if not file_path:
            return

        try:
            self.current_pdf_path = file_path
            self.doc = fitz.open(file_path)
            self.total_pages = len(self.doc)
            self.current_page_num = 0

            # Initialize data for this PDF if needed
            if file_path not in self.pdf_data:
                self.pdf_data[file_path] = {}

            self.load_page()
            self.app.update_status(f"Opened {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open PDF: {str(e)}")

    def load_page(self):
        """Load the current page of the PDF"""
        if not self.current_pdf_path or not hasattr(self, "doc"):
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

        # Load any existing boxes for this page
        self.current_boxes = []
        if self.current_page_num in self.pdf_data.get(self.current_pdf_path, {}):
            self.current_boxes = self.pdf_data[self.current_pdf_path][
                self.current_page_num
            ]
            self.draw_boxes()

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

    def run_ocr(self):
        """Run OCR on the current page"""
        if not self.current_image:
            messagebox.showinfo("Info", "Please open a PDF first")
            return

        self.app.update_status("Running OCR...")

        try:
            # Convert PIL image to OpenCV image (RGB to BGR)
            img_cv = np.array(self.current_image)
            img_cv = img_cv[:, :, ::-1].copy()

            # Convert to grayscale for OCR
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # Use Tesseract to get text and bounding boxes
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

            # Process OCR results
            self.current_boxes = []
            for i in range(len(data["text"])):
                # Skip empty text
                if not data["text"][i].strip():
                    continue

                # Get bounding box
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]

                # Add to current boxes
                self.current_boxes.append([x, y, x + w, y + h, data["text"][i], "O"])

            # Store OCR results
            if self.current_pdf_path not in self.pdf_data:
                self.pdf_data[self.current_pdf_path] = {}

            self.pdf_data[self.current_pdf_path][
                self.current_page_num
            ] = self.current_boxes

            # Draw boxes
            self.draw_boxes()

            self.app.update_status(
                f"OCR completed: {len(self.current_boxes)} text boxes detected"
            )
        except Exception as e:
            messagebox.showerror("Error", f"OCR failed: {str(e)}")
            self.app.update_status("OCR failed")

    def draw_boxes(self):
        """Draw bounding boxes on the canvas"""
        # Clear existing boxes
        self.canvas.delete("box")

        for i, box in enumerate(self.current_boxes):
            x0, y0, x1, y1, text, label = box

            # Determine color based on selection status and label
            if i == self.selected_box_idx:
                outline = "red"
                width = 2
            elif label != "O":
                outline = "green"
                width = 2
            else:
                outline = "blue"
                width = 1

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

    def on_canvas_click(self, event):
        """Handle canvas click event"""
        if not self.current_boxes:
            return

        # Get canvas coordinates
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Check if click is inside any box
        for i, box in enumerate(self.current_boxes):
            x0, y0, x1, y1, _, _ = box
            if x0 <= x <= x1 and y0 <= y <= y1:
                # Select this box
                self.selected_box_idx = i

                # Update label combobox
                self.label_var.set(box[5])

                self.app.update_status(f"Selected box: {box[4][:30]}")

                self.draw_boxes()
                return

        # If click is not inside any box, deselect
        self.selected_box_idx = -1
        self.draw_boxes()

    def add_label(self):
        """Add a new label"""
        new_label = simpledialog.askstring("New Label", "Enter new label name:")
        if new_label and new_label not in self.labels:
            self.labels.append(new_label)
            self.label2id[new_label] = len(self.labels) - 1
            self.id2label[len(self.labels) - 1] = new_label
            self.label_combo["values"] = tuple(self.labels)

    def apply_label(self):
        """Apply selected label to selected box"""
        if self.selected_box_idx >= 0:
            label = self.label_var.get()
            if label:
                # Update the box's label
                self.current_boxes[self.selected_box_idx][5] = label

                # Update in storage
                if (
                    self.current_pdf_path in self.pdf_data
                    and self.current_page_num in self.pdf_data[self.current_pdf_path]
                ):
                    self.pdf_data[self.current_pdf_path][
                        self.current_page_num
                    ] = self.current_boxes

                self.draw_boxes()
                self.app.update_status(f"Applied label '{label}' to selected box")

    def delete_box(self):
        """Delete selected box"""
        if self.selected_box_idx >= 0:
            # Remove the box
            del self.current_boxes[self.selected_box_idx]

            # Update in storage
            if (
                self.current_pdf_path in self.pdf_data
                and self.current_page_num in self.pdf_data[self.current_pdf_path]
            ):
                self.pdf_data[self.current_pdf_path][
                    self.current_page_num
                ] = self.current_boxes

            self.selected_box_idx = -1
            self.draw_boxes()
            self.app.update_status("Box deleted")

    def save_annotations(self):
        """Save annotations to a file"""
        if not self.pdf_data:
            messagebox.showinfo("Info", "No annotations to save")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Annotations",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if not file_path:
            return

        try:
            save_annotations(
                self.pdf_data, self.labels, self.label2id, self.id2label, file_path
            )
            self.app.update_status(f"Annotations saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")

    def load_annotations(self):
        """Load annotations from a file"""
        file_path = filedialog.askopenfilename(
            title="Load Annotations",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if not file_path:
            return

        try:
            pdf_data, label_info = load_annotations(file_path)

            # Update labels
            if "labels" in label_info:
                self.labels = label_info["labels"]
                self.label_combo["values"] = tuple(self.labels)

            if "label2id" in label_info:
                self.label2id = label_info["label2id"]

            if "id2label" in label_info:
                self.id2label = label_info["id2label"]

            # Update PDF data
            self.pdf_data = pdf_data

            # Reload current page if needed
            if self.current_pdf_path in self.pdf_data:
                self.load_page()

            self.app.update_status(f"Annotations loaded from {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load annotations: {str(e)}")
