
Exported Document Dataset
------------------------

This directory contains labeled document data exported from PDF OCR Tool.

Contents:
- labeled_data.json: JSON file containing all text boxes
- images/: Directory containing document page images

JSON Format:
[
  {
    "id": "document_filename_pagenum",
    "image": "path/to/image.png",
    "width": original_image_width,
    "height": original_image_height,
    "boxes": [
      {
        "text": "text content of box",
        "bbox": [x, y, width, height],  // Normalized to 0-1000 range
        "bbox_pixels": [x, y, width, height],  // Original pixel coordinates
        "label": "label name"
      },
      ...
    ]
  },
  ...
]

Example usage in Python:
```python
import json
import os
from PIL import Image

# Load the dataset
with open("labeled_data.json", "r") as f:
    dataset = json.load(f)

# Process each page
for page in dataset:
    page_id = page["id"]
    image_path = os.path.join(os.path.dirname("labeled_data.json"), page["image"])
    image = Image.open(image_path)
    
    # Process each text box
    for box in page["boxes"]:
        text = box["text"]
        # Use normalized coordinates
        x, y, w, h = box["bbox"]
        # Or use pixel coordinates
        x_px, y_px, w_px, h_px = box["bbox_pixels"]
        label = box["label"]
        
        print(f"Page {page_id}: Label '{label}' for text '{text}' at {x},{y},{w},{h} (normalized)")
```
