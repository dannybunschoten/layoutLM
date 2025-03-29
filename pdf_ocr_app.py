import os
import sys
import json
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import fitz  # PyMuPDF
import numpy as np
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import cv2
import pytesseract
from typing import List, Dict, Tuple, Optional


class OCRAnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LayoutLM PDF OCR Annotation Tool")
        self.root.geometry("1200x800")

        # Application state
        self.current_pdf_path = None
        self.current_page_num = 0
        self.total_pages = 0
        self.current_image = None
        self.current_boxes = []  # [x0, y0, x1, y1, text, label]
        self.selected_box_idx = -1
        self.combining_boxes = False
        self.boxes_to_combine = []
        self.pdf_data = {}  # {pdf_path: {page_num: [boxes]}}

        # LayoutLM components
        self.processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base", apply_ocr=False
        )
        self.model = None
        self.labels = []
        self.id2label = {}
        self.label2id = {}

        # Create UI components
        self._setup_ui()

    def _setup_ui(self):
        # Main frame layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel (controls)
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
        self.label_combo = ttk.Combobox(label_frame, textvariable=self.label_var)
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

        combine_frame = ttk.Frame(annot_frame)
        combine_frame.pack(fill=tk.X, padx=5, pady=2)
        self.combine_btn = ttk.Button(
            combine_frame, text="Start Combine", command=self.toggle_combine_mode
        )
        self.combine_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.finish_combine_btn = ttk.Button(
            combine_frame, text="Finish", command=self.finish_combine, state=tk.DISABLED
        )
        self.finish_combine_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Training and inference
        train_frame = ttk.LabelFrame(left_frame, text="Training & Inference")
        train_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            train_frame, text="Save Annotations", command=self.save_annotations
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            train_frame, text="Load Annotations", command=self.load_annotations
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(train_frame, text="Train Model", command=self.train_model).pack(
            fill=tk.X, padx=5, pady=2
        )
        ttk.Button(train_frame, text="Run Inference", command=self.run_inference).pack(
            fill=tk.X, padx=5, pady=2
        )

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

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def open_pdf(self):
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
            self.status_var.set(f"Opened {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open PDF: {str(e)}")

    def load_page(self):
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
        if hasattr(self, "doc") and self.current_page_num > 0:
            self.current_page_num -= 1
            self.load_page()

    def next_page(self):
        if hasattr(self, "doc") and self.current_page_num < self.total_pages - 1:
            self.current_page_num += 1
            self.load_page()

    def run_ocr(self):
        if not self.current_image:
            messagebox.showinfo("Info", "Please open a PDF first")
            return

        self.status_var.set("Running OCR...")
        self.root.update()

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
                self.current_boxes.append(
                    [x, y, x + w, y + h, data["text"][i], "UNKNOWN"]
                )

            # Store OCR results
            if self.current_pdf_path not in self.pdf_data:
                self.pdf_data[self.current_pdf_path] = {}

            self.pdf_data[self.current_pdf_path][
                self.current_page_num
            ] = self.current_boxes

            # Draw boxes
            self.draw_boxes()

            self.status_var.set(
                f"OCR completed: {len(self.current_boxes)} text boxes detected"
            )
        except Exception as e:
            messagebox.showerror("Error", f"OCR failed: {str(e)}")
            self.status_var.set("OCR failed")

    def draw_boxes(self):
        # Clear existing boxes
        self.canvas.delete("box")

        for i, box in enumerate(self.current_boxes):
            x0, y0, x1, y1, text, label = box

            # Determine color based on selection status and label
            if i == self.selected_box_idx:
                outline = "red"
                width = 2
            elif label != "UNKNOWN":
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
        if not self.current_boxes:
            return

        # Get canvas coordinates
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Check if click is inside any box
        for i, box in enumerate(self.current_boxes):
            x0, y0, x1, y1, _, _ = box
            if x0 <= x <= x1 and y0 <= y <= y1:
                if self.combining_boxes:
                    # Add to combine list
                    if i in self.boxes_to_combine:
                        self.boxes_to_combine.remove(i)
                    else:
                        self.boxes_to_combine.append(i)

                    self.status_var.set(
                        f"Selected {len(self.boxes_to_combine)} boxes for combining"
                    )
                else:
                    # Select this box
                    self.selected_box_idx = i

                    # Update label combobox
                    self.label_var.set(box[5])

                    self.status_var.set(f"Selected box: {box[4][:30]}")

                self.draw_boxes()
                return

        # If click is not inside any box, deselect
        self.selected_box_idx = -1
        self.draw_boxes()

    def add_label(self):
        new_label = simpledialog.askstring("New Label", "Enter new label name:")
        if new_label and new_label not in self.labels:
            self.labels.append(new_label)
            self.label2id[new_label] = len(self.labels)
            self.id2label[len(self.labels)] = new_label
            self.label_combo["values"] = tuple(self.labels)

    def apply_label(self):
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
                self.status_var.set(f"Applied label '{label}' to selected box")

    def delete_box(self):
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
            self.status_var.set("Box deleted")

    def toggle_combine_mode(self):
        self.combining_boxes = not self.combining_boxes
        if self.combining_boxes:
            self.combine_btn.config(text="Cancel Combine")
            self.finish_combine_btn.config(state=tk.NORMAL)
            self.boxes_to_combine = []
            self.status_var.set("Combine mode activated. Select boxes to combine.")
        else:
            self.combine_btn.config(text="Start Combine")
            self.finish_combine_btn.config(state=tk.DISABLED)
            self.boxes_to_combine = []
            self.status_var.set("Combine mode canceled")

    def finish_combine(self):
        if len(self.boxes_to_combine) < 2:
            messagebox.showinfo("Info", "Please select at least 2 boxes to combine")
            return

        # Sort the indices to maintain order
        self.boxes_to_combine.sort()

        # Get the boxes
        boxes_to_combine = [self.current_boxes[i] for i in self.boxes_to_combine]

        # Calculate the new bounding box (min x0, min y0, max x1, max y1)
        x0 = min(box[0] for box in boxes_to_combine)
        y0 = min(box[1] for box in boxes_to_combine)
        x1 = max(box[2] for box in boxes_to_combine)
        y1 = max(box[3] for box in boxes_to_combine)

        # Combine text with spaces in between
        text = " ".join(box[4] for box in boxes_to_combine)

        # Use the label of the first box, or "UNKNOWN" if no label
        label = (
            boxes_to_combine[0][5] if boxes_to_combine[0][5] != "UNKNOWN" else "UNKNOWN"
        )

        # Create the new combined box
        new_box = [x0, y0, x1, y1, text, label]

        # Remove the original boxes (in reverse order to maintain indices)
        for i in sorted(self.boxes_to_combine, reverse=True):
            del self.current_boxes[i]

        # Add the new combined box
        self.current_boxes.append(new_box)

        # Update in storage
        if (
            self.current_pdf_path in self.pdf_data
            and self.current_page_num in self.pdf_data[self.current_pdf_path]
        ):
            self.pdf_data[self.current_pdf_path][
                self.current_page_num
            ] = self.current_boxes

        # Exit combine mode
        self.combining_boxes = False
        self.combine_btn.config(text="Start Combine")
        self.finish_combine_btn.config(state=tk.DISABLED)
        self.boxes_to_combine = []

        # Redraw boxes
        self.draw_boxes()
        self.status_var.set("Boxes combined successfully")

    def save_annotations(self):
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
            # Convert the data to a serializable format
            serializable_data = {}
            for pdf_path, pages in self.pdf_data.items():
                pdf_name = os.path.basename(pdf_path)
                serializable_data[pdf_name] = {}
                for page_num, boxes in pages.items():
                    serializable_data[pdf_name][str(page_num)] = boxes

            # Save labels
            serializable_data["labels"] = self.labels

            # Ensure label2id and id2label are in the correct format
            # Convert all keys to strings for JSON serialization
            serializable_data["label2id"] = {k: v for k, v in self.label2id.items()}
            serializable_data["id2label"] = {
                str(k): v for k, v in self.id2label.items()
            }

            with open(file_path, "w") as f:
                json.dump(serializable_data, f, indent=2)

            self.status_var.set(f"Annotations saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")

    def load_annotations(self):
        file_path = filedialog.askopenfilename(
            title="Load Annotations",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if not file_path:
            return

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Load labels
            if "labels" in data:
                self.labels = data["labels"]

                # Recreate label2id and id2label with continuous indices starting from 0
                self.label2id = {}
                self.id2label = {}

                # First add "O" label if it doesn't exist
                if "O" not in self.labels:
                    self.labels.append("O")

                # Create mappings with continuous indices
                for i, label in enumerate(self.labels):
                    self.label2id[label] = i
                    self.id2label[i] = label

                # Update combo box
                self.label_combo["values"] = tuple(self.labels)

            # Remove labels from data
            data_copy = data.copy()
            if "labels" in data_copy:
                del data_copy["labels"]
            if "label2id" in data_copy:
                del data_copy["label2id"]
            if "id2label" in data_copy:
                del data_copy["id2label"]

            # Convert the data back to the internal format
            self.pdf_data = {}
            for pdf_name, pages in data_copy.items():
                # Find full path for this PDF (if open)
                full_path = None
                if (
                    self.current_pdf_path
                    and os.path.basename(self.current_pdf_path) == pdf_name
                ):
                    full_path = self.current_pdf_path
                else:
                    # Try to find the PDF in the current directory
                    possible_path = os.path.join(os.path.dirname(file_path), pdf_name)
                    if os.path.exists(possible_path):
                        full_path = possible_path

                if full_path:
                    self.pdf_data[full_path] = {}
                    for page_num, boxes in pages.items():
                        self.pdf_data[full_path][int(page_num)] = boxes

            # Reload current page if needed
            if self.current_pdf_path in self.pdf_data:
                self.load_page()

            self.status_var.set(f"Annotations loaded from {file_path}")

            # Debug info
            print("Loaded labels:", self.labels)
            print("label2id:", self.label2id)
            print("id2label:", self.id2label)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load annotations: {str(e)}")
            import traceback

            traceback.print_exc()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load annotations: {str(e)}")

    def prepare_training_data(self):
        if not self.pdf_data:
            messagebox.showinfo("Info", "No annotations available for training")
            return None

        # Ensure we have the "O" label for outside/background elements
        if "O" not in self.label2id:
            # Add "O" label with ID 0
            # First, recreate label mappings to be continuous
            new_labels = ["O"] + [label for label in self.labels if label != "O"]
            new_label2id = {label: i for i, label in enumerate(new_labels)}
            new_id2label = {i: label for i, label in enumerate(new_labels)}

            self.labels = new_labels
            self.label2id = new_label2id
            self.id2label = new_id2label

            # Update combobox values
            self.label_combo["values"] = tuple(self.labels)

        # Check if we have enough labeled data
        labeled_boxes = []
        for pdf_path, pages in self.pdf_data.items():
            for page_num, boxes in pages.items():
                for box in boxes:
                    if box[5] != "UNKNOWN":
                        labeled_boxes.append(box)

        if len(labeled_boxes) < 5:  # Lower threshold for testing
            messagebox.showinfo(
                "Info",
                f"Only {len(labeled_boxes)} labeled boxes found. Please label more data.",
            )
            return None

        # Create a list to hold all training examples
        training_examples = []

        # Process each PDF and page
        for pdf_path, pages in self.pdf_data.items():
            try:
                doc = fitz.open(pdf_path)

                for page_num_str, boxes in pages.items():
                    # Convert page_num to integer if it's a string
                    page_num = (
                        int(page_num_str)
                        if isinstance(page_num_str, str)
                        else page_num_str
                    )

                    # Get only labeled boxes
                    labeled_boxes = [box for box in boxes if box[5] != "UNKNOWN"]

                    if not labeled_boxes:
                        continue

                    # Get the page and render to image
                    page = doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    # Process the page through LayoutLM processor
                    words = []
                    word_boxes = []
                    word_labels = []

                    for box in labeled_boxes:
                        x0, y0, x1, y1, text, label = box

                        # Skip empty text
                        if not text.strip():
                            continue

                        words.append(text)

                        # Normalize coordinates to 0-1000 range for LayoutLM
                        # Ensure coordinates are within bounds
                        x0 = max(0, min(x0, img.width))
                        y0 = max(0, min(y0, img.height))
                        x1 = max(0, min(x1, img.width))
                        y1 = max(0, min(y1, img.height))

                        normalized_box = [
                            int(1000 * x0 / img.width),
                            int(1000 * y0 / img.height),
                            int(1000 * x1 / img.width),
                            int(1000 * y1 / img.height),
                        ]

                        # Validate box coordinates
                        if not all(0 <= coord <= 1000 for coord in normalized_box):
                            print(f"Invalid box coordinates: {normalized_box}")
                            # Skip this box
                            words.pop()
                            continue

                        word_boxes.append(normalized_box)

                        # Convert label to id, defaulting to "O" (0)
                        label_id = self.label2id.get(label, self.label2id.get("O", 0))
                        word_labels.append(label_id)

                    # Ensure we have data after filtering
                    if not words:
                        continue

                    # Ensure all arrays have the same length
                    min_len = min(len(words), len(word_boxes), len(word_labels))
                    words = words[:min_len]
                    word_boxes = word_boxes[:min_len]
                    word_labels = word_labels[:min_len]

                    # Add to training examples
                    training_examples.append(
                        {
                            "image": img,
                            "words": words,
                            "boxes": word_boxes,
                            "labels": word_labels,
                        }
                    )

                doc.close()
            except Exception as e:
                print(f"Error processing PDF {pdf_path}: {str(e)}")
                import traceback

                traceback.print_exc()

        if not training_examples:
            messagebox.showinfo("Info", "No valid training examples found")
            return None

        print(f"Prepared {len(training_examples)} training examples")
        for i, example in enumerate(training_examples):
            print(
                f"Example {i}: {len(example['words'])} words, {len(example['boxes'])} boxes, {len(example['labels'])} labels"
            )
            if len(example["words"]) > 0:
                print(f"  First word: {example['words'][0]}")
                print(f"  First box: {example['boxes'][0]}")
                print(f"  First label: {example['labels'][0]}")

        return training_examples

    def train_model(self):
        # Prepare training data
        training_examples = self.prepare_training_data()

        if not training_examples:
            return

        # Convert to dataset format for Hugging Face
        try:
            dataset_dict = {
                "image": [example["image"] for example in training_examples],
                "words": [example["words"] for example in training_examples],
                "boxes": [example["boxes"] for example in training_examples],
                "labels": [example["labels"] for example in training_examples],
            }

            # Create a small wrapper for the dataset just to check its structure
            for key, value in dataset_dict.items():
                if key != "image":  # Skip image to avoid large output
                    print(f"{key}: {value}")

            dataset = Dataset.from_dict(dataset_dict)

            print(f"Created dataset with {len(dataset)} examples")
            print(f"Dataset features: {dataset.features}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare dataset: {str(e)}")
            self.status_var.set("Dataset preparation failed")
            import traceback

            traceback.print_exc()
            return

        # Get the number of labels from the dictionary
        num_labels = len(self.id2label)
        print(f"Number of labels: {num_labels}")
        print(f"Label mapping: {self.id2label}")

        # First, let's test the processor with a simple example
        test_example = {
            "image": dataset_dict["image"][0],
            "words": dataset_dict["words"][0],
            "boxes": dataset_dict["boxes"][0],
            "labels": dataset_dict["labels"][0],
        }

        try:
            print("Testing processor with first example...")
            test_output = self.processor(
                test_example["image"],
                text=[test_example["words"]],
                boxes=[test_example["boxes"]],
                word_labels=[test_example["labels"]],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            print("Processor test successful!")
        except Exception as e:
            print(f"Processor test failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return  # Stop if test fails

        # Manual processing for now to avoid dataset.map issues
        print("Manually processing dataset...")
        processed_data = {
            "input_ids": [],
            "attention_mask": [],
            "bbox": [],
            "labels": [],
        }

        for i in range(len(dataset)):
            try:
                # Process one example at a time
                example = {
                    "image": dataset[i]["image"],
                    "words": dataset[i]["words"],
                    "boxes": dataset[i]["boxes"],
                    "labels": dataset[i]["labels"],
                }

                # Process through LayoutLM
                processed = self.processor(
                    example["image"],
                    text=[example["words"]],
                    boxes=[example["boxes"]],
                    word_labels=[example["labels"]],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                )

                # Add to processed data
                for key in processed_data:
                    processed_data[key].append(processed[key].squeeze().cpu())

                print(f"Processed example {i+1}/{len(dataset)}")
            except Exception as e:
                print(f"Error processing example {i}: {str(e)}")
                import traceback

                traceback.print_exc()
                # Skip this example

        if not processed_data["input_ids"]:
            messagebox.showerror("Error", "Failed to process any examples")
            self.status_var.set("Training failed")
            return

        # Stack tensors to create a batch
        try:
            print("Stacking processed data...")
            for key in processed_data:
                processed_data[key] = torch.stack(processed_data[key])
            print("Stacking successful!")
        except Exception as e:
            print(f"Error stacking tensors: {str(e)}")
            messagebox.showerror("Error", f"Failed to stack tensors: {str(e)}")
            self.status_var.set("Training failed")
            return

        # Load pretrained model
        # First, make sure id2label and label2id are properly formatted
        label2id = {
            k: int(v) if isinstance(v, str) else v for k, v in self.label2id.items()
        }
        id2label = {
            int(k) if isinstance(k, str) else k: v for k, v in self.id2label.items()
        }

        # Verify num_labels matches the length of id2label
        num_labels = len(id2label)

        # Now initialize the model with consistent parameters
        try:
            print(f"Loading model with {num_labels} labels...")
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                "microsoft/layoutlmv3-base",
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Training failed")
            return

        # Instead of using the Trainer, implement a simple training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.model.to(device)

        # Define optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        # Create DataLoader - we'll handle batching manually
        batch_size = 1
        num_examples = processed_data["input_ids"].shape[0]

        # Number of training epochs
        num_epochs = 10

        # Train the model
        self.status_var.set("Training the model... This may take a while")
        self.root.update()

        try:
            # Use a try-except block to catch any training errors
            print("Starting training...")
            self.model.train()

            for epoch in range(num_epochs):
                epoch_loss = 0
                # Process examples one by one
                for i in range(0, num_examples, batch_size):
                    # Get batch
                    batch_end = min(i + batch_size, num_examples)

                    # Get input tensors
                    input_ids = processed_data["input_ids"][i:batch_end].to(device)
                    attention_mask = processed_data["attention_mask"][i:batch_end].to(
                        device
                    )
                    bbox = processed_data["bbox"][i:batch_end].to(device)
                    labels = processed_data["labels"][i:batch_end].to(device)

                    # Clear gradients
                    optimizer.zero_grad()

                    # Forward pass with explicitly named parameters
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        bbox=bbox,
                        labels=labels,
                    )

                    loss = outputs.loss

                    # Backward pass
                    loss.backward()

                    # Update weights
                    optimizer.step()

                    epoch_loss += loss.item()

                    # Update status
                    self.status_var.set(
                        f"Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size+1}/{(num_examples+batch_size-1)//batch_size}, Loss: {loss.item():.4f}"
                    )
                    self.root.update()

                # Print epoch stats
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/((num_examples+batch_size-1)//batch_size):.4f}"
                )

            # Save the model and processor
            os.makedirs("./layoutlm_model", exist_ok=True)
            self.model.save_pretrained("./layoutlm_model")
            self.processor.tokenizer.save_pretrained("./layoutlm_model")

            # Save label mappings
            with open("./layoutlm_model/label_mapping.json", "w") as f:
                import json

                json.dump(
                    {
                        "labels": self.labels,
                        "label2id": self.label2id,
                        "id2label": {str(k): v for k, v in self.id2label.items()},
                    },
                    f,
                )

            self.status_var.set(
                "Model training completed and saved to ./layoutlm_model"
            )
            messagebox.showinfo("Success", "Model training completed successfully!")
            print("Training completed successfully!")
        except Exception as e:
            # Print the full stack trace for debugging
            import traceback

            traceback.print_exc()

            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.status_var.set("Training failed")
            print(f"Training failed: {str(e)}")

    def train_model(self):
        # Prepare training data
        training_examples = self.prepare_training_data()

        if not training_examples:
            return

        # Convert to dataset format for Hugging Face
        try:
            dataset_dict = {
                "image": [example["image"] for example in training_examples],
                "words": [example["words"] for example in training_examples],
                "boxes": [example["boxes"] for example in training_examples],
                "labels": [example["labels"] for example in training_examples],
            }

            # Create a small wrapper for the dataset just to check its structure
            for key, value in dataset_dict.items():
                if key != "image":  # Skip image to avoid large output
                    print(f"{key}: {value}")

            dataset = Dataset.from_dict(dataset_dict)

            print(f"Created dataset with {len(dataset)} examples")
            print(f"Dataset features: {dataset.features}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare dataset: {str(e)}")
            self.status_var.set("Dataset preparation failed")
            import traceback

            traceback.print_exc()
            return

        # Get the number of labels from the dictionary
        num_labels = len(self.id2label)
        print(f"Number of labels: {num_labels}")
        print(f"Label mapping: {self.id2label}")

        # First, let's test the processor with a simple example
        test_example = {
            "image": dataset_dict["image"][0],
            "words": dataset_dict["words"][0],
            "boxes": dataset_dict["boxes"][0],
            "labels": dataset_dict["labels"][0],
        }

        try:
            print("Testing processor with first example...")
            test_output = self.processor(
                test_example["image"],
                text=[test_example["words"]],
                boxes=[test_example["boxes"]],
                word_labels=[test_example["labels"]],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            print("Processor test successful!")
        except Exception as e:
            print(f"Processor test failed: {str(e)}")
            import traceback

            traceback.print_exc()
            # We'll continue anyway and hope the batched processing works

        # Manual processing for now to avoid dataset.map issues
        print("Manually processing dataset...")
        processed_data = {
            "input_ids": [],
            "attention_mask": [],
            "bbox": [],
            "labels": [],
        }

        for i in range(len(dataset)):
            try:
                # Process one example at a time
                example = {
                    "image": dataset[i]["image"],
                    "words": dataset[i]["words"],
                    "boxes": dataset[i]["boxes"],
                    "labels": dataset[i]["labels"],
                }

                # Process through LayoutLM
                processed = self.processor(
                    example["image"],
                    text=[example["words"]],
                    boxes=[example["boxes"]],
                    word_labels=[example["labels"]],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                )

                # Add to processed data
                for key in processed_data:
                    processed_data[key].append(processed[key].squeeze().cpu())

                print(f"Processed example {i+1}/{len(dataset)}")
            except Exception as e:
                print(f"Error processing example {i}: {str(e)}")
                # Skip this example

        if not processed_data["input_ids"]:
            messagebox.showerror("Error", "Failed to process any examples")
            self.status_var.set("Training failed")
            return

        # Stack tensors to create a batch
        try:
            print("Stacking processed data...")
            for key in processed_data:
                processed_data[key] = torch.stack(processed_data[key])
            print("Stacking successful!")
        except Exception as e:
            print(f"Error stacking tensors: {str(e)}")
            messagebox.showerror("Error", f"Failed to stack tensors: {str(e)}")
            self.status_var.set("Training failed")
            return

        # Create TensorDataset
        tensor_dataset = torch.utils.data.TensorDataset(
            processed_data["input_ids"],
            processed_data["attention_mask"],
            processed_data["bbox"],
            processed_data["labels"],
        )

        # Load pretrained model
        # First, make sure id2label and label2id are properly formatted
        label2id = {
            k: int(v) if isinstance(v, str) else v for k, v in self.label2id.items()
        }
        id2label = {
            int(k) if isinstance(k, str) else k: v for k, v in self.id2label.items()
        }

        # Verify num_labels matches the length of id2label
        num_labels = len(id2label)

        # Now initialize the model with consistent parameters
        try:
            print(f"Loading model with {num_labels} labels...")
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                "microsoft/layoutlmv3-base",
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Training failed")
            return

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./layoutlm_model",
            per_device_train_batch_size=1,  # Small batch size for stability
            gradient_accumulation_steps=4,  # Accumulate gradients
            num_train_epochs=3,
            learning_rate=5e-5,
            save_strategy="epoch",
            evaluation_strategy="no",  # No evaluation to simplify
            logging_dir="./logs",
            logging_steps=10,
            load_best_model_at_end=False,
            save_total_limit=1,  # Only keep most recent checkpoint
            fp16=False,  # Disable mixed precision to avoid issues
            dataloader_drop_last=False,
            report_to="none",  # Disable wandb and other trackers
        )

        # Define custom trainer since we're using TensorDataset
        class CustomTrainer(Trainer):
            def get_train_dataloader(self):
                return torch.utils.data.DataLoader(
                    tensor_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    shuffle=True,
                )

        # Initialize trainer
        trainer = CustomTrainer(
            model=self.model, args=training_args, tokenizer=self.processor.tokenizer
        )

        # Train the model
        self.status_var.set("Training the model... This may take a while")
        self.root.update()

        try:
            # Use a try-except block to catch any training errors
            print("Starting training...")
            trainer.train()

            # Save the model and processor
            self.model.save_pretrained("./layoutlm_model")
            self.processor.tokenizer.save_pretrained("./layoutlm_model")

            # Save label mappings
            with open("./layoutlm_model/label_mapping.json", "w") as f:
                import json

                json.dump(
                    {
                        "labels": self.labels,
                        "label2id": self.label2id,
                        "id2label": {str(k): v for k, v in self.id2label.items()},
                    },
                    f,
                )

            self.status_var.set(
                "Model training completed and saved to ./layoutlm_model"
            )
            messagebox.showinfo("Success", "Model training completed successfully!")
            print("Training completed successfully!")
        except Exception as e:
            # Print the full stack trace for debugging
            import traceback

            traceback.print_exc()

            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.status_var.set("Training failed")
            print(f"Training failed: {str(e)}")

        # Define a preprocessing function that ensures consistent handling
        def preprocess_data(examples):
            try:
                # Debug output
                print("Processing batch with:")
                print(f"Number of examples: {len(examples['words'])}")
                for i in range(len(examples["words"])):
                    print(
                        f"Example {i}: {len(examples['words'][i])} words, {len(examples['boxes'][i])} boxes, {len(examples['labels'][i])} labels"
                    )

                # Process each example to ensure consistency
                for i in range(len(examples["words"])):
                    # Check if words and boxes match in length
                    words_len = len(examples["words"][i])
                    boxes_len = len(examples["boxes"][i])
                    labels_len = len(examples["labels"][i])

                    # Make sure all arrays have the same length
                    if not (words_len == boxes_len == labels_len):
                        print(
                            f"Inconsistent lengths in example {i}: words={words_len}, boxes={boxes_len}, labels={labels_len}"
                        )

                        # Take the minimum length of all arrays
                        min_len = min(words_len, boxes_len, labels_len)
                        examples["words"][i] = examples["words"][i][:min_len]
                        examples["boxes"][i] = examples["boxes"][i][:min_len]
                        examples["labels"][i] = examples["labels"][i][:min_len]

                # Create a dummy example for debugging first
                dummy_examples = {
                    "image": examples["image"][0:1],
                    "words": [["test"]],
                    "boxes": [[[0, 0, 100, 100]]],
                    "labels": [[0]],
                }

                # Try processing with the dummy example first
                dummy_result = self.processor(
                    dummy_examples["image"],
                    text=dummy_examples["words"],
                    boxes=dummy_examples["boxes"],
                    word_labels=dummy_examples["labels"],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                )

                print("Dummy preprocessing succeeded, now trying with real examples")

                # Now try with the real examples
                processed_examples = self.processor(
                    examples["image"],
                    text=examples["words"],
                    boxes=examples["boxes"],
                    word_labels=examples["labels"],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                )

                return processed_examples

            except Exception as e:
                # If there's an error, print detailed information for debugging
                print(f"Error processing examples: {str(e)}")
                import traceback

                traceback.print_exc()

                # Example diagnostic info
                if len(examples["words"]) > 0:
                    print("\nExample data for debugging:")
                    print(f"Number of words: {len(examples['words'][0])}")
                    if len(examples["words"][0]) > 0:
                        print(f"First word: {examples['words'][0][0]}")

                    if len(examples["boxes"]) > 0 and len(examples["boxes"][0]) > 0:
                        print(f"First box: {examples['boxes'][0][0]}")

                    if len(examples["labels"]) > 0 and len(examples["labels"][0]) > 0:
                        print(f"First label: {examples['labels'][0][0]}")
                        print(f"Label type: {type(examples['labels'][0][0])}")

                # Create a minimal valid output to continue
                return {
                    "input_ids": torch.zeros(
                        (len(examples["image"]), 512), dtype=torch.long
                    ),
                    "attention_mask": torch.zeros(
                        (len(examples["image"]), 512), dtype=torch.long
                    ),
                    "bbox": torch.zeros(
                        (len(examples["image"]), 512, 4), dtype=torch.long
                    ),
                    "labels": torch.zeros(
                        (len(examples["image"]), 512), dtype=torch.long
                    ),
                }

        # Process dataset
        processed_dataset = dataset.map(
            preprocess_data, batched=True, remove_columns=["image", "words", "boxes"]
        )

        # Load pretrained model
        # First, make sure id2label and label2id are properly formatted
        # The error suggests the mapping starts at 1, but we need it to start at 0
        # Recreate proper mappings
        label2id = {}
        id2label = {}

        # First add the "O" (Outside) label with index 0
        label2id["O"] = 0
        id2label[0] = "O"

        # Then add all other labels starting from index 1
        for i, label in enumerate(self.labels, start=1):
            label2id[label] = i
            id2label[i] = label

        # Update internal state
        self.label2id = label2id
        self.id2label = id2label

        # Verify num_labels matches the length of id2label
        num_labels = len(id2label)

        # Now initialize the model with consistent parameters
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./layoutlm_model",
            per_device_train_batch_size=1,  # Reduce batch size for stability
            gradient_accumulation_steps=4,  # Accumulate gradients to compensate for small batch size
            num_train_epochs=3,
            learning_rate=5e-5,
            save_strategy="epoch",
            evaluation_strategy="no",  # No evaluation to simplify training
            logging_dir="./logs",
            logging_steps=10,
            load_best_model_at_end=False,  # Since we don't have evaluation
            save_total_limit=1,  # Only keep the most recent checkpoint
            fp16=False,  # Disable mixed precision to avoid issues
            dataloader_drop_last=False,  # Keep all examples
            report_to="none",  # Disable wandb and other trackers
        )

        # Define trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset,
            tokenizer=self.processor.tokenizer,  # Provide tokenizer to handle padding correctly
        )

        # Train the model
        self.status_var.set("Training the model... This may take a while")
        self.root.update()

        try:
            # Use a try-except block to catch any training errors
            trainer.train()

            # Save the model and processor
            self.model.save_pretrained("./layoutlm_model")
            self.processor.tokenizer.save_pretrained("./layoutlm_model")

            # Save label mappings
            with open("./layoutlm_model/label_mapping.json", "w") as f:
                import json

                json.dump(
                    {
                        "labels": self.labels,
                        "label2id": self.label2id,
                        "id2label": {str(k): v for k, v in self.id2label.items()},
                    },
                    f,
                )

            self.status_var.set(
                "Model training completed and saved to ./layoutlm_model"
            )
            messagebox.showinfo("Success", "Model training completed successfully!")
        except Exception as e:
            # Print the full stack trace for debugging
            import traceback

            traceback.print_exc()

            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.status_var.set("Training failed")

    def run_inference(self):
        if not self.model:
            messagebox.showinfo("Info", "Please train a model first")
            return

        # Ask user to select a PDF for inference
        file_path = filedialog.askopenfilename(
            title="Select PDF for inference",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )

        if not file_path:
            return

        try:
            # Open the PDF
            doc = fitz.open(file_path)

            # Process each page
            results = {}
            for page_num in range(len(doc)):
                self.status_var.set(f"Processing page {page_num+1}/{len(doc)}")
                self.root.update()

                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Run OCR
                img_cv = np.array(img)
                img_cv = img_cv[:, :, ::-1].copy()
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

                # Use Tesseract to get text and bounding boxes
                data = pytesseract.image_to_data(
                    gray, output_type=pytesseract.Output.DICT
                )

                # Process OCR results
                words = []
                word_boxes = []
                original_boxes = []

                for i in range(len(data["text"])):
                    # Skip empty text
                    if not data["text"][i].strip():
                        continue

                    # Get text and bounding box
                    text = data["text"][i]
                    x = data["left"][i]
                    y = data["top"][i]
                    w = data["width"][i]
                    h = data["height"][i]

                    # Skip invalid boxes (e.g., zero width/height)
                    if w <= 0 or h <= 0:
                        continue

                    # Save original box coordinates
                    original_boxes.append([x, y, x + w, y + h, text])

                    # Normalize coordinates for LayoutLM (0-1000 range)
                    norm_box = [
                        int(1000 * x / img.width),
                        int(1000 * y / img.height),
                        int(1000 * (x + w) / img.width),
                        int(1000 * (y + h) / img.height),
                    ]

                    words.append(text)
                    word_boxes.append(norm_box)

                # If no words detected, continue to next page
                if not words:
                    continue

                try:
                    # Process through LayoutLM
                    inputs = self.processor(
                        images=img,
                        text=[words],
                        boxes=[word_boxes],
                        return_tensors="pt",
                    )

                    # Run inference
                    with torch.no_grad():
                        outputs = self.model(**inputs)

                    # Get predictions
                    predictions = outputs.logits.argmax(-1).squeeze().tolist()

                    # If predictions is a single number, convert to list
                    if isinstance(predictions, int):
                        predictions = [predictions]

                    # Map token predictions back to words
                    word_preds = []
                    token_ids = inputs.input_ids.squeeze().tolist()

                    # Get special token IDs
                    cls_token_id = self.processor.tokenizer.cls_token_id
                    sep_token_id = self.processor.tokenizer.sep_token_id
                    pad_token_id = self.processor.tokenizer.pad_token_id

                    # Track which word we're on
                    current_word_idx = 0

                    # For each token in the sequence
                    for i, token_id in enumerate(token_ids):
                        # Skip special tokens (CLS, SEP, PAD)
                        if token_id in [cls_token_id, sep_token_id, pad_token_id]:
                            continue

                        # If we've processed all words, break
                        if current_word_idx >= len(words):
                            break

                        # Get the prediction for this token
                        if i < len(predictions):
                            pred = predictions[i]
                            label = self.id2label.get(pred, "UNKNOWN")

                            # Add to word_preds if this is the first token of the word
                            if current_word_idx == len(word_preds):
                                word_preds.append(label)

                            # If this token starts with ## (continuation of word), skip to next token
                            # without incrementing current_word_idx
                            decoded_token = self.processor.tokenizer.decode([token_id])
                            if not decoded_token.startswith("##"):
                                current_word_idx += 1

                    # If we have fewer predictions than words, pad with "UNKNOWN"
                    while len(word_preds) < len(words):
                        word_preds.append("UNKNOWN")

                    # Store results for this page
                    page_results = []
                    for i, box in enumerate(original_boxes):
                        if i < len(word_preds):
                            # Add the predicted label to the box
                            box.append(word_preds[i])
                            page_results.append(box)

                    results[page_num] = page_results

                except Exception as e:
                    print(f"Error processing page {page_num}: {str(e)}")
                    import traceback

                    traceback.print_exc()

            # Store results in PDF data
            self.pdf_data[file_path] = results

            # Switch to the inference PDF
            self.current_pdf_path = file_path
            self.doc = doc
            self.total_pages = len(doc)
            self.current_page_num = 0

            # Load the first page if there are results
            if 0 in results:
                self.current_boxes = results[0]
            else:
                self.current_boxes = []

            self.load_page()

            self.status_var.set(f"Inference completed on {os.path.basename(file_path)}")
            messagebox.showinfo("Success", "Inference completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Inference failed: {str(e)}")
            self.status_var.set("Inference failed")
            import traceback

            traceback.print_exc()
