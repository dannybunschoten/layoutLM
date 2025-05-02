
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
print(f"\nVisualizing page {page_idx}: {page['id']}")

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
print("\nVisualizing with original pixel coordinates")

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
print("\nRun with a different page index: python load_dataset.py PAGE_INDEX")
