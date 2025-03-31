"""
Inference logic for PDF information extraction
"""

import os
import json
import torch
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import fitz  # PyMuPDF

from ..config import Config
from ..data.preprocessing import (
    extract_text_from_pdf_page,
    extract_features_for_inference,
    map_predictions_to_words,
)


class PDFExtractorInference:
    def __init__(self, config: Config, processor, model=None):
        """
        Initialize inference for PDF information extraction

        Args:
            config: Configuration object
            processor: LayoutLM processor for feature extraction
            model: Pre-trained model (optional)
        """
        self.config = config
        self.processor = processor
        self.model = model
        self.id2label = {}
        self.label2id = {}

        # Load label mappings if model provided
        if model and hasattr(model, "config"):
            self.id2label = model.config.id2label
            self.label2id = model.config.label2id

    def load_label_mapping(self, mapping_path: str) -> None:
        """
        Load label mapping from a JSON file

        Args:
            mapping_path: Path to label mapping JSON file
        """
        with open(mapping_path, "r") as f:
            mapping = json.load(f)

        if "id2label" in mapping:
            self.id2label = {int(k): v for k, v in mapping["id2label"].items()}

        if "label2id" in mapping:
            self.label2id = mapping["label2id"]

    def process_pdf(self, pdf_path: str) -> Dict[int, List[List]]:
        """
        Process a PDF file and extract information

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary mapping page numbers to lists of annotated boxes
        """
        if not self.model:
            raise ValueError("Model not loaded. Please load a model first.")

        # Ensure device is set properly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Process PDF
        results = {}
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            print(f"Processing page {page_num+1}/{len(doc)}...")

            # Extract text boxes
            page_image, boxes = extract_text_from_pdf_page(pdf_path, page_num)

            # Skip if no boxes detected
            if not boxes:
                continue

            # Extract features
            inputs = extract_features_for_inference(page_image, boxes, self.processor)

            # Move tensors to device
            inputs = {key: val.to(device) for key, val in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get predictions
            predictions = outputs.logits.argmax(-1).squeeze().tolist()

            # Convert to list if single value
            if isinstance(predictions, int):
                predictions = [predictions]

            # Map predictions back to words
            words = [box[4] for box in boxes]
            token_ids = inputs["input_ids"].squeeze().tolist()

            word_predictions = map_predictions_to_words(
                predictions, token_ids, words, self.id2label, self.processor.tokenizer
            )

            # Update boxes with predicted labels
            annotated_boxes = []
            for i, (box, pred) in enumerate(zip(boxes, word_predictions)):
                if i < len(word_predictions):
                    x0, y0, x1, y1, text, _ = box
                    annotated_boxes.append([x0, y0, x1, y1, text, pred])

            results[page_num] = annotated_boxes

        doc.close()
        return results

    def extract_structured_info(
        self, annotated_boxes: List[List], target_labels: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Extract structured information from annotated boxes

        Args:
            annotated_boxes: List of boxes with predicted labels
            target_labels: List of labels to extract (if None, extract all)

        Returns:
            Dictionary mapping labels to lists of text
        """
        if target_labels is None:
            # Extract all unique labels from boxes
            target_labels = set(
                box[5]
                for box in annotated_boxes
                if box[5] != "O" and box[5] != "UNKNOWN"
            )
            target_labels = list(target_labels)

        # Initialize result dictionary
        result = {label: [] for label in target_labels}

        # Extract text for each label
        for box in annotated_boxes:
            _, _, _, _, text, label = box
            if label in target_labels:
                result[label].append(text)

        return result
