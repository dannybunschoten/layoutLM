"""
Dataset creation and processing for PDF information extraction
"""

import os
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import torch
from PIL import Image
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
import fitz  # PyMuPDF

from ..config import Config


class PDFDataset:
    def __init__(self, config: Config, processor):
        """
        Initialize PDF dataset handler

        Args:
            config: Configuration object
            processor: LayoutLM processor for feature extraction
        """
        self.config = config
        self.processor = processor
        self.label_list = []
        self.id2label = {}
        self.label2id = {}

    def setup_labels(self, labels: List[str]) -> None:
        """
        Set up label mappings

        Args:
            labels: List of label strings
        """
        # Make sure 'O' is included and is first (index 0)
        if "O" not in labels:
            labels = ["O"] + [label for label in labels if label != "O"]
        else:
            # Make sure 'O' is first
            labels.remove("O")
            labels = ["O"] + labels

        self.label_list = labels
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.label2id = {label: i for i, label in enumerate(labels)}

    def load_annotations(self, annotations: Dict) -> None:
        """
        Load annotations from dictionary

        Args:
            annotations: Dictionary containing annotations
        """
        if "labels" in annotations:
            self.setup_labels(annotations["labels"])

        if "label2id" in annotations:
            self.label2id = annotations["label2id"]

        if "id2label" in annotations:
            # Convert string keys to integers if needed
            self.id2label = {
                int(k) if isinstance(k, str) else k: v
                for k, v in annotations["id2label"].items()
            }

    def create_dataset_from_annotations(
        self, pdf_data: Dict[str, Dict[int, List]]
    ) -> Tuple[Dataset, Dict[str, Any]]:
        """
        Create a dataset from PDF annotations

        Args:
            pdf_data: Dictionary of format {pdf_path: {page_num: [boxes]}}
                      where boxes are [x0, y0, x1, y1, text, label]

        Returns:
            dataset: Hugging Face dataset
            features_info: Dictionary with dataset feature information
        """
        examples = []

        for pdf_path, pages in pdf_data.items():
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

                    # Extract text, boxes, and labels
                    words = []
                    bboxes = []
                    ner_tags = []

                    for box in labeled_boxes:
                        x0, y0, x1, y1, text, label = box

                        # Skip empty text
                        if not text.strip():
                            continue

                        words.append(text)

                        # Normalize coordinates to 0-1000 range for LayoutLM
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

                        bboxes.append(normalized_box)

                        # Convert label to id, defaulting to "O" (0)
                        label_id = self.label2id.get(label, self.label2id.get("O", 0))
                        ner_tags.append(label_id)

                    # Ensure we have data after filtering
                    if not words:
                        continue

                    # Create example
                    example = {
                        "id": f"{os.path.basename(pdf_path)}_{page_num}",
                        "tokens": words,
                        "bboxes": bboxes,
                        "ner_tags": ner_tags,
                        "image": img,
                    }

                    examples.append(example)

                doc.close()
            except Exception as e:
                print(f"Error processing PDF {pdf_path}: {str(e)}")
                import traceback

                traceback.print_exc()

        if not examples:
            raise ValueError("No valid examples found in the annotations")

        # Create dataset features
        features = Features(
            {
                "id": Value(dtype="string"),
                "tokens": Sequence(feature=Value(dtype="string")),
                "bboxes": Sequence(
                    feature=Sequence(feature=Value(dtype="int64"), length=4)
                ),
                "ner_tags": Sequence(
                    feature=ClassLabel(
                        num_classes=len(self.id2label),
                        names=list(self.id2label.values()),
                    )
                ),
                "image": Array3D(dtype="uint8", shape=(None, None, 3)),
            }
        )

        # Create dataset
        dataset = Dataset.from_dict(
            {
                "id": [example["id"] for example in examples],
                "tokens": [example["tokens"] for example in examples],
                "bboxes": [example["bboxes"] for example in examples],
                "ner_tags": [example["ner_tags"] for example in examples],
                "image": [np.array(example["image"]) for example in examples],
            },
            features=features,
        )

        # Return dataset and feature info
        return dataset, {
            "label_list": self.label_list,
            "id2label": self.id2label,
            "label2id": self.label2id,
            "num_labels": len(self.id2label),
        }

    def prepare_features(self, examples):
        """
        Prepare features for the model using the processor

        Args:
            examples: Dictionary of examples

        Returns:
            Dictionary of processed features
        """
        images = examples["image"]
        words = examples["tokens"]
        boxes = examples["bboxes"]
        word_labels = examples["ner_tags"]

        encoding = self.processor(
            images,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=self.config.model["max_seq_length"],
            return_tensors="pt",
        )

        # Convert batch output (first dim is batch) to list of individual examples
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bbox": encoding["bbox"].squeeze(0),
            "labels": encoding["labels"].squeeze(0),
        }

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """
        Process dataset to prepare it for training

        Args:
            dataset: Raw dataset

        Returns:
            Processed dataset ready for training
        """
        # Define custom features for proper formatting
        processed_features = Features(
            {
                "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
                "input_ids": Sequence(feature=Value(dtype="int64")),
                "attention_mask": Sequence(Value(dtype="int64")),
                "bbox": Array2D(dtype="int64", shape=(512, 4)),
                "labels": Sequence(feature=Value(dtype="int64")),
            }
        )

        # Process the dataset
        processed_dataset = dataset.map(
            self.prepare_features,
            batched=True,
            batch_size=1,  # Process one example at a time for stability
            remove_columns=dataset.column_names,
            features=processed_features,
        )

        # Set format to PyTorch tensors
        processed_dataset.set_format("torch")

        return processed_dataset
