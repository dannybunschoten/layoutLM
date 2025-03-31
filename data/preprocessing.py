"""
Data preprocessing functions for PDF information extraction
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import torch
from PIL import Image
import cv2
import pytesseract
import fitz  # PyMuPDF


def extract_text_from_pdf_page(
    pdf_path: str, page_num: int, scale: float = 2.0
) -> Tuple[Image.Image, List[List]]:
    """
    Extract text and bounding boxes from a PDF page using OCR

    Args:
        pdf_path: Path to PDF file
        page_num: Page number to process
        scale: Scale factor for rendering PDF pages

    Returns:
        tuple: (PIL Image of page, list of text boxes [x0, y0, x1, y1, text, "UNKNOWN"])
    """
    # Open the PDF and get the page
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # Render page to image
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Convert to OpenCV format for OCR
    img_cv = np.array(img)
    img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR

    # Convert to grayscale for OCR
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Use Tesseract to get text and bounding boxes
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    # Process OCR results
    boxes = []
    for i in range(len(data["text"])):
        # Skip empty text
        if not data["text"][i].strip():
            continue

        # Get bounding box
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        # Skip invalid boxes (e.g., zero width/height)
        if w <= 0 or h <= 0:
            continue

        # Add to boxes with "UNKNOWN" label
        boxes.append([x, y, x + w, y + h, data["text"][i], "UNKNOWN"])

    doc.close()
    return img, boxes


def normalize_boxes(
    boxes: List[List], image_width: int, image_height: int
) -> List[List]:
    """
    Normalize bounding box coordinates to 0-1000 range for LayoutLM

    Args:
        boxes: List of bounding boxes [x0, y0, x1, y1, text, label]
        image_width: Width of the image
        image_height: Height of the image

    Returns:
        List of boxes with normalized coordinates
    """
    normalized_boxes = []
    for box in boxes:
        x0, y0, x1, y1, text, label = box

        # Ensure coordinates are within bounds
        x0 = max(0, min(x0, image_width))
        y0 = max(0, min(y0, image_height))
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))

        # Normalize to 0-1000 range
        normalized_box = [
            int(1000 * x0 / image_width),
            int(1000 * y0 / image_height),
            int(1000 * x1 / image_width),
            int(1000 * y1 / image_height),
        ]

        # Skip invalid boxes
        if not all(0 <= coord <= 1000 for coord in normalized_box):
            continue

        normalized_boxes.append(normalized_box + [text, label])

    return normalized_boxes


def extract_features_for_inference(
    image: Image.Image, boxes: List[List], processor
) -> Dict[str, torch.Tensor]:
    """
    Extract features for inference using the processor

    Args:
        image: PIL Image of the page
        boxes: List of bounding boxes [x0, y0, x1, y1, text]
        processor: LayoutLM processor

    Returns:
        Dictionary of processed features
    """
    words = [box[4] for box in boxes]
    word_boxes = []

    # Normalize box coordinates
    for box in boxes:
        x0, y0, x1, y1, _ = box

        # Normalize to 0-1000 range
        normalized_box = [
            int(1000 * x0 / image.width),
            int(1000 * y0 / image.height),
            int(1000 * x1 / image.width),
            int(1000 * y1 / image.height),
        ]

        word_boxes.append(normalized_box)

    # Process through LayoutLM processor
    inputs = processor(
        images=image,
        text=[words],
        boxes=[word_boxes],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )

    return inputs


def map_predictions_to_words(
    predictions: List[int],
    token_ids: List[int],
    words: List[str],
    id2label: Dict[int, str],
    tokenizer,
) -> List[str]:
    """
    Map token-level predictions back to word-level predictions

    Args:
        predictions: List of token-level predictions
        token_ids: List of token IDs
        words: List of words
        id2label: Mapping from label IDs to label strings
        tokenizer: Tokenizer used

    Returns:
        List of word-level predictions
    """
    # Initialize prediction list
    word_preds = []

    # Get special token IDs
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    # Track which word we're on
    current_word_idx = 0

    # For each token in the sequence
    for i, token_id in enumerate(token_ids):
        # Skip special tokens
        if token_id in [cls_token_id, sep_token_id, pad_token_id]:
            continue

        # If we've processed all words, break
        if current_word_idx >= len(words):
            break

        # Get the prediction for this token
        if i < len(predictions):
            pred = predictions[i]
            label = id2label.get(pred, "UNKNOWN")

            # Add to word_preds if this is the first token of the word
            if current_word_idx == len(word_preds):
                word_preds.append(label)

            # Check if this is a continuation token
            decoded_token = tokenizer.decode([token_id])
            if not decoded_token.startswith("##"):
                current_word_idx += 1

    # Pad with "UNKNOWN" if needed
    while len(word_preds) < len(words):
        word_preds.append("UNKNOWN")

    return word_preds
