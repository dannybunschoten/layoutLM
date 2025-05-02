import os
import json
from PIL import Image

class SimpleExporter:
    """Simple utility class for exporting labeled data to a basic JSON format"""
    
    @staticmethod
    def export_dataset(documents, output_dir, normalize_boxes=True, include_unlabeled=True):
        """
        Export labeled documents to a simple JSON format
        
        Args:
            documents: List of Document objects with labeled text boxes
            output_dir: Directory to save the dataset
            normalize_boxes: Whether to normalize bounding boxes to 0-1000 range
            include_unlabeled: Whether to include boxes with "O" label
            
        Returns:
            Path to the exported JSON file
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Prepare dataset structure
        dataset = []
        
        for doc_idx, doc in enumerate(documents):
            for page_idx, boxes in enumerate(doc.page_boxes):
                if not boxes:  # Skip pages with no boxes
                    continue
                
                # Create a unique ID for this page
                page_id = f"{doc.filename.replace('.pdf', '')}_{page_idx}"
                
                # Save the image
                image = doc.images[page_idx]
                image_path = os.path.join(images_dir, f"{page_id}.png")
                
                # Get image dimensions for normalization
                width, height = image.size if hasattr(image, 'size') else (image.width, image.height)
                
                # Convert to RGB if needed and save as PNG
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(image_path, "PNG")
                
                # Get relative path for JSON storage
                rel_image_path = os.path.join("images", f"{page_id}.png")
                
                # Collect all text boxes
                page_boxes = []
                
                for box in boxes:
                    # Include all boxes if include_unlabeled is True, otherwise only include labeled boxes
                    if include_unlabeled or box.label != "O":
                        # Original bbox coordinates
                        orig_bbox = [box.x, box.y, box.w, box.h]
                        
                        # Normalized bbox coordinates (0-1000 range)
                        if normalize_boxes:
                            # Convert [x, y, w, h] to normalized [x, y, w, h]
                            norm_bbox = [
                                int(1000 * box.x / width),
                                int(1000 * box.y / height),
                                int(1000 * box.w / width),
                                int(1000 * box.h / height)
                            ]
                        else:
                            norm_bbox = orig_bbox
                        
                        page_boxes.append({
                            "text": " ".join(box.words),
                            "bbox": norm_bbox,
                            "bbox_pixels": orig_bbox,  # Include original pixel coordinates for reference
                            "label": box.label
                        })
                
                # Create page entry
                page_entry = {
                    "id": page_id,
                    "image": rel_image_path,
                    "width": width,
                    "height": height,
                    "boxes": page_boxes
                }
                
                dataset.append(page_entry)
        
        # Write JSON file
        json_path = os.path.join(output_dir, "labeled_data.json")
        with open(json_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Create a simple readme
        readme_path = os.path.join(output_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write("""
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
""")
        
        return json_path

    @staticmethod
    def create_loader_script(output_dir):
        """
        Create a simple Python script to load and visualize the dataset
        
        Args:
            output_dir: Directory where the dataset is saved
            
        Returns:
            Path to the script
        """
        script_path = os.path.join(output_dir, "load_dataset.py")
        
        with open(script_path, 'w') as f:
            f.write("""
import json
import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Load the dataset
json_path = os.path.join(os.path.dirname(__file__), "labeled_data.json")
with open(json_path, "r") as f:
    dataset = json.load(f)

print(f"Loaded dataset with {len(dataset)} document pages")

# Print dataset statistics
all_labels = set()
total_boxes = 0
for page in dataset:
    for box in page["boxes"]:
        all_labels.add(box["label"])
        total_boxes += 1

print(f"Total boxes: {total_boxes}")
print(f"Unique labels: {sorted(list(all_labels))}")

# Choose a page to visualize (default to first page)
page_idx = 0
if len(sys.argv) > 1:
    try:
        page_idx = int(sys.argv[1])
    except:
        print(f"Invalid page index: {sys.argv[1]}, using 0 instead")

if page_idx >= len(dataset):
    print(f"Page index {page_idx} out of range, using 0 instead")
    page_idx = 0

# Visualize the selected page
page = dataset[page_idx]
print(f"\\nVisualizing page {page_idx}: {page['id']}")

# Open the image
image_path = os.path.join(os.path.dirname(json_path), page["image"])
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

# Try to get a font, or use default if not available
try:
    font = ImageFont.truetype("arial.ttf", 12)
except:
    font = ImageFont.load_default()

# Define colors for different labels
label_colors = {}
import random
random.seed(42)  # For consistent colors

# Add default color for 'O' label
label_colors["O"] = (200, 200, 200)  # Light gray

for box in page["boxes"]:
    if box["label"] not in label_colors:
        # Generate a random color
        r = random.randint(50, 200)
        g = random.randint(50, 200)
        b = random.randint(50, 200)
        label_colors[box["label"]] = (r, g, b)

# Use pixel coordinates for visualization
print("\\nVisualizing with original pixel coordinates")

# Draw boxes on the image
for box in page["boxes"]:
    x, y, w, h = box["bbox_pixels"]
    label = box["label"]
    color = label_colors[label]
    
    # Draw rectangle
    draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
    
    # Draw label (skip for 'O' to reduce clutter)
    if label != "O":
        draw.text((x, y-15), label, fill=color, font=font)

# Save the visualization
output_path = os.path.join(os.path.dirname(json_path), f"visualization_{page_idx}.png")
image.save(output_path)
print(f"Visualization saved to: {output_path}")
print("\\nRun with a different page index: python load_dataset.py PAGE_INDEX")
""")
        
        return script_path