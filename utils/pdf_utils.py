"""
PDF processing utilities
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont


def save_annotations(
    pdf_data: Dict[str, Dict[int, List]],
    labels: List[str],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    output_path: str,
) -> None:
    """
    Save PDF annotations to a JSON file

    Args:
        pdf_data: Dictionary of format {pdf_path: {page_num: [boxes]}}
        labels: List of label strings
        label2id: Mapping from label strings to label IDs
        id2label: Mapping from label IDs to label strings
        output_path: Path to output JSON file
    """
    # Convert the data to a serializable format
    serializable_data = {}
    for pdf_path, pages in pdf_data.items():
        pdf_name = os.path.basename(pdf_path)
        serializable_data[pdf_name] = {}
        for page_num, boxes in pages.items():
            serializable_data[pdf_name][str(page_num)] = boxes

    # Add label information
    serializable_data["labels"] = labels
    serializable_data["label2id"] = label2id
    serializable_data["id2label"] = {str(k): v for k, v in id2label.items()}

    # Save to file
    with open(output_path, "w") as f:
        json.dump(serializable_data, f, indent=2)


def load_annotations(
    file_path: str,
) -> Tuple[Dict[str, Dict[int, List]], Dict[str, Any]]:
    """
    Load annotations from a JSON file

    Args:
        file_path: Path to JSON file

    Returns:
        tuple: (pdf_data, label_info)
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract label information
    label_info = {}
    if "labels" in data:
        label_info["labels"] = data["labels"]

    if "label2id" in data:
        label_info["label2id"] = data["label2id"]

    if "id2label" in data:
        label_info["id2label"] = {int(k): v for k, v in data["id2label"].items()}

    # Remove label info from data copy
    data_copy = data.copy()
    for key in ["labels", "label2id", "id2label"]:
        if key in data_copy:
            del data_copy[key]

    # Convert data to internal format
    pdf_data = {}
    for pdf_name, pages in data_copy.items():
        # Find full path for this PDF if it exists in the same directory
        possible_path = os.path.join(os.path.dirname(file_path), pdf_name)
        full_path = possible_path if os.path.exists(possible_path) else pdf_name

        pdf_data[full_path] = {}
        for page_num, boxes in pages.items():
            pdf_data[full_path][int(page_num)] = boxes

    return pdf_data, label_info


def visualize_annotations(
    image: Image.Image, boxes: List[List], label_colors: Optional[Dict[str, str]] = None
) -> Image.Image:
    """
    Visualize annotations on an image

    Args:
        image: PIL Image
        boxes: List of boxes [x0, y0, x1, y1, text, label]
        label_colors: Dictionary mapping labels to colors

    Returns:
        Annotated image
    """
    # Create a copy of the image
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Default font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    # Default colors if not provided
    if label_colors is None:
        label_colors = {"UNKNOWN": "blue", "O": "gray"}

    # Draw boxes and labels
    for box in boxes:
        x0, y0, x1, y1, text, label = box

        # Determine color
        color = label_colors.get(label, "green")

        # Draw rectangle
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        # Draw label and text
        draw.text(
            (x0, y0 - 15),
            f"{label}: {text[:20]}{'...' if len(text) > 20 else ''}",
            fill=color,
            font=font,
        )

    return img_copy


def extract_pages_from_pdf(
    pdf_path: str, output_dir: str, scale: float = 2.0
) -> List[str]:
    """
    Extract pages from a PDF as images

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save images
        scale: Scale factor for rendering PDF pages

    Returns:
        List of paths to extracted images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open PDF
    doc = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Extract pages
    image_paths = []
    for page_num, page in enumerate(doc):
        # Render page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))

        # Save image
        image_path = os.path.join(output_dir, f"{pdf_name}_page_{page_num+1}.png")
        pix.save(image_path)

        image_paths.append(image_path)

    doc.close()
    return image_paths
