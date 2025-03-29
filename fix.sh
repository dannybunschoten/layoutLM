#!/bin/bash

# Create a fix script to add the missing methods
cat > fix_canvas_methods.py << 'EOF'
import re

# Read the app.py file
with open('form_annotator/app.py', 'r') as f:
    content = f.read()

# Add the missing on_canvas_drag method if it doesn't exist
if 'def on_canvas_drag(self, event):' not in content:
    # Find where to add the method (after on_canvas_click)
    pattern = r'def on_canvas_click\(self, event\):(.*?)(?=\n    def )'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        on_canvas_click_method = match.group(0)
        insert_pos = match.end()
        
        # Define the missing methods
        missing_methods = """
    def on_canvas_drag(self, event):
        """Handle mouse drag on canvas - update annotation rectangle."""
        if not self.annotation_mode or not self.current_annotation:
            return
        
        # Update rectangle end coordinates
        self.current_annotation["end_x"] = self.canvas.canvasx(event.x)
        self.current_annotation["end_y"] = self.canvas.canvasy(event.y)
        
        # Update rectangle display
        self.canvas.coords(
            self.current_rect,
            self.current_annotation["start_x"],
            self.current_annotation["start_y"],
            self.current_annotation["end_x"],
            self.current_annotation["end_y"]
        )

    def on_canvas_release(self, event):
        """Handle mouse release - finalize annotation."""
        if not self.annotation_mode or not self.current_annotation:
            return
        
        # Finalize the annotation coordinates
        x1 = min(self.current_annotation["start_x"], self.current_annotation["end_x"])
        y1 = min(self.current_annotation["start_y"], self.current_annotation["end_y"])
        x2 = max(self.current_annotation["start_x"], self.current_annotation["end_x"])
        y2 = max(self.current_annotation["start_y"], self.current_annotation["end_y"])
        
        # Check if the rectangle is too small (likely a click rather than drag)
        if (x2 - x1 < 10) or (y2 - y1 < 10):
            self.canvas.delete("new_annotation")
            self.current_annotation = None
            return
        
        # Create normalized box coordinates (0-1000 scale for LayoutLM)
        page = self.pdf_document[self.current_page]
        width, height = page.rect.width, page.rect.height
        scale = self.scale_factor * (self.zoom_level / 100)  # Account for zoom level
        
        # Normalize to 0-1000 range for LayoutLM
        bbox = [
            int(x1 / (width * scale) * 1000),
            int(y1 / (height * scale) * 1000),
            int(x2 / (width * scale) * 1000),
            int(y2 / (height * scale) * 1000)
        ]
        
        # Validate field name
        field_name = self.field_name.get().strip()
        if not field_name:
            messagebox.showwarning("Missing Information", "Please enter a field name.")
            self.canvas.delete("new_annotation")
            self.current_annotation = None
            return
        
        # Store annotation data
        pdf_path = self.pdf_document.name
        
        if self.current_page not in self.annotations[pdf_path]:
            self.annotations[pdf_path][self.current_page] = []
        
        # Check for field name uniqueness
        for page_num, page_annotations in self.annotations[pdf_path].items():
            for existing_annotation in page_annotations:
                if existing_annotation.name == field_name:
                    messagebox.showwarning(
                        "Duplicate Field", 
                        f"Field '{field_name}' already exists on page {page_num+1}. Please use a unique name."
                    )
                    self.canvas.delete("new_annotation")
                    self.current_annotation = None
                    return
        
        # Create annotation object
        annotation = FormFieldAnnotation(
            name=field_name,
            field_type=self.field_type.get(),
            bbox=bbox,
            page=self.current_page,
            display={
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            }
        )
        
        self.annotations[pdf_path][self.current_page].append(annotation)
        
        # Add to tree view
        self.annotations_tree.insert("", tk.END, values=(
            annotation.name, 
            annotation.field_type, 
            self.current_page + 1
        ))
        
        # Reset current annotation
        self.current_annotation = None
        
        # Redraw all annotations
        self.draw_annotations()
        
        # If collecting training data, prompt for field value
        if self.is_collecting_training_data:
            self.prompt_for_field_value(annotation)
"""
        
        # Insert the missing methods
        new_content = content[:insert_pos] + missing_methods + content[insert_pos:]
        
        # Write the updated content back to the file
        with open('form_annotator/app.py', 'w') as f:
            f.write(new_content)
        
        print("Added missing canvas methods: on_canvas_drag and on_canvas_release")
    else:
        print("Could not find appropriate location to add methods")
else:
    print("Methods already exist, no changes made")

EOF

# Run the fix script
python3 fix_canvas_methods.py

# Reinstall the package
pip install -e .

echo "Now you can run the application with: form-annotator"