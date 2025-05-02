import os
from typing import Optional, Tuple
import numpy as np
import cv2
import pytesseract
from pdf2image import convert_from_path
from PyQt5.QtGui import QPixmap, QImage
from models.text_box import TextBox

class Document:
    """Class representing a PDF document with OCR data"""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.images = []  # List of page images
        self.page_boxes = []  # List of lists, one list of TextBox objects per page
        self.current_page = 0
        self.thumbnail = None  # Thumbnail of the first page
        self.load_document()

    def load_document(self) -> bool:
        """Load the PDF and convert pages to images"""
        try:
            self.images = convert_from_path(self.file_path)
            # Initialize empty box lists for each page
            self.page_boxes = [[] for _ in range(len(self.images))]
            # Create thumbnail of first page
            if self.images:
                self.create_thumbnail()
            return True
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            return False

    def create_thumbnail(self, size : Tuple[int, int]=(200, 200)) -> None:
        """Create a thumbnail of the first page"""
        if not self.images:
            return

        first_page = self.images[0]
        img_np = np.array(first_page)

        # Calculate scaling to fit within thumbnail size while maintaining aspect ratio
        h, w, _ = img_np.shape
        scale = min(size[0] / w, size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize the image
        thumbnail = cv2.resize(img_np, (new_w, new_h))

        # Convert to QPixmap for display
        rgb_image = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_image.shape
        bytes_per_line = 3 * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.thumbnail = QPixmap.fromImage(q_img)

    def run_ocr_on_page(self, page_idx: int) -> bool:
        """Run OCR on a specific page"""
        if page_idx >= len(self.images) or page_idx < 0:
            return False

        try:
            # Get the page image
            pil_image = self.images[page_idx]
            image = np.array(pil_image)
            # Convert RGB to BGR (OpenCV format)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Run OCR with pytesseract
            ocr_data = pytesseract.image_to_data(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                output_type=pytesseract.Output.DICT,
            )

            # Clear existing boxes for this page
            self.page_boxes[page_idx] = []

            # Create TextBox objects directly from OCR data
            for i in range(len(ocr_data["text"])):
                if ocr_data["text"][i].strip():
                    x1 = ocr_data["left"][i]
                    y1 = ocr_data["top"][i]
                    x2 = ocr_data["left"][i] + ocr_data["width"][i]
                    y2 = ocr_data["top"][i] + ocr_data["height"][i]
                    words = [ocr_data["text"][i]]
                    label = "O"  # Default label

                    # Create TextBox and add it to the page's boxes
                    box = TextBox(x1, y1, x2, y2, words, label)
                    self.page_boxes[page_idx].append(box)

            return True
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return False

    def get_page_count(self) -> int:
        """Return the number of pages in the document"""
        return len(self.images)

    def get_current_page_image(self) -> np.ndarray:
        """Get the current page image with boxes drawn on it"""
        if not self.images or self.current_page >= len(self.images):
            return None

        # Get current image
        pil_image = self.images[self.current_page]
        img_np = np.array(pil_image)

        return img_np

    def get_box_count(self, page_idx: Optional[int] = None) -> int:
        """Get the number of boxes on a page or in the entire document"""
        if page_idx is not None:
            if page_idx < len(self.page_boxes):
                return len(self.page_boxes[page_idx])
            return 0
        else:
            return sum(len(boxes) for boxes in self.page_boxes)