import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QLabel,
    QWidget,
    QScrollArea,
    QToolTip,
    QComboBox,
    QInputDialog,
    QLineEdit,
    QListWidget,
    QSplitter,
    QDialog,
    QMessageBox,
)
from PyQt5.QtGui import QPixmap, QCursor, QImage
from PyQt5.QtCore import Qt, QTimer, QSize

from models.text_box import TextBox
from ui.model_training import ModelTrainingDialog
from ui.model_prediction import ModelPredictionDialog

from models.document import Document
from ui.document_list_item import DocumentListItem

class PDFOCRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.documents = []  # List of Document objects
        self.current_document_idx = -1  # Index of currently selected document
        self.selected_indices = []  # Track the selection order of boxes
        self.available_labels = ["O"]  # List of available labels
        self.text_boxes = []  # Current page's TextBox objects
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Multiple Document OCR Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        # Create a splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Document list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Document list
        self.doc_list = QListWidget()
        self.doc_list.setIconSize(QSize(160, 90))
        self.doc_list.itemClicked.connect(self.on_document_selected)
        left_layout.addWidget(QLabel("Documents:"))
        left_layout.addWidget(self.doc_list)

        # Upload button
        self.upload_btn = QPushButton("Upload PDF")
        self.upload_btn.clicked.connect(self.upload_pdf)
        left_layout.addWidget(self.upload_btn)

        # Document info
        self.doc_info_label = QLabel("No document selected")
        left_layout.addWidget(self.doc_info_label)

        # Right panel - OCR viewer and controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # OCR controls
        control_layout = QHBoxLayout()

        # OCR button
        self.ocr_btn = QPushButton("Run OCR")
        self.ocr_btn.clicked.connect(self.run_ocr)
        self.ocr_btn.setEnabled(False)
        control_layout.addWidget(self.ocr_btn)

        # Navigation buttons
        self.prev_btn = QPushButton("Previous Page")
        self.prev_btn.clicked.connect(self.prev_page)
        self.prev_btn.setEnabled(False)
        control_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next Page")
        self.next_btn.clicked.connect(self.next_page)
        self.next_btn.setEnabled(False)
        control_layout.addWidget(self.next_btn)

        # Combine button
        self.combine_btn = QPushButton("Combine Selected Boxes")
        self.combine_btn.clicked.connect(self.combine_selected_boxes)
        self.combine_btn.setEnabled(False)
        control_layout.addWidget(self.combine_btn)

        # Clear selection button
        self.clear_sel_btn = QPushButton("Clear Selection")
        self.clear_sel_btn.clicked.connect(self.clear_selection)
        self.clear_sel_btn.setEnabled(False)
        control_layout.addWidget(self.clear_sel_btn)

        right_layout.addLayout(control_layout)

        # Label controls
        label_layout = QHBoxLayout()

        self.label_selector = QComboBox()
        self.label_selector.addItems(self.available_labels)
        self.label_selector.setEnabled(False)
        self.label_selector.currentIndexChanged.connect(self.apply_label_to_selection)
        label_layout.addWidget(QLabel("Label:"))
        label_layout.addWidget(self.label_selector)

        self.add_label_btn = QPushButton("+")
        self.add_label_btn.setMaximumWidth(30)
        self.add_label_btn.clicked.connect(self.add_new_label)
        label_layout.addWidget(self.add_label_btn)

        right_layout.addLayout(label_layout)

        # Status and page info
        info_layout = QHBoxLayout()
        self.status_label = QLabel("Upload a PDF to begin")
        self.page_label = QLabel("Page: 0/0")
        info_layout.addWidget(self.status_label)
        info_layout.addWidget(self.page_label)
        right_layout.addLayout(info_layout)

        # Create scroll area for the image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        # Create image label with mouse tracking
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.image_label.mouseMoveEvent = self.on_image_hover
        self.image_label.mousePressEvent = self.on_image_click
        self.scroll_area.setWidget(self.image_label)

        # Variables for tooltip handling
        self.tooltip_timer = QTimer()
        self.tooltip_timer.setSingleShot(True)
        self.tooltip_timer.timeout.connect(self.show_text_tooltip)
        self.hover_position = None
        self.hover_text = None

        right_layout.addWidget(self.scroll_area)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        # Set initial sizes (30% for document list, 70% for OCR view)
        splitter.setSizes([int(self.width() * 0.3), int(self.width() * 0.7)])

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        self.setCentralWidget(central_widget)

        self.train_model_btn = QPushButton("Train Model")
        self.train_model_btn.clicked.connect(self.show_model_training)
        control_layout.addWidget(self.train_model_btn)
        
        self.predict_btn = QPushButton("Run Prediction")
        self.predict_btn.clicked.connect(self.show_model_prediction)
        self.predict_btn.setEnabled(False)
        control_layout.addWidget(self.predict_btn)


    def upload_pdf(self):
        """Upload a new PDF document"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF File", "", "PDF Files (*.pdf)", options=options
        )

        if file_path:
            # Check if document already exists in the list
            for doc in self.documents:
                if doc.file_path == file_path:
                    self.status_label.setText(
                        f"Document already loaded: {os.path.basename(file_path)}"
                    )
                    return

            self.status_label.setText(f"Loading: {os.path.basename(file_path)}")

            # Create new document
            document = Document(file_path)
            if document.get_page_count() > 0:
                # Add document to list
                self.documents.append(document)

                # Add document to list widget
                item = DocumentListItem(document)
                self.doc_list.addItem(item)

                # Select the newly added document
                self.doc_list.setCurrentItem(item)
                self.on_document_selected(item)

                self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")
            else:
                self.status_label.setText(
                    f"Error loading document: {os.path.basename(file_path)}"
                )

    def on_document_selected(self, item):
        """Handle document selection from the list"""
        if not item:
            return

        # Find the document index
        for i, doc in enumerate(self.documents):
            if doc.file_path == item.document.file_path:
                self.current_document_idx = i
                break

        # Clear current selection
        self.selected_indices.clear()

        # Update document info
        self.update_document_info()

        # Display the current page
        self.display_current_page()

        # Enable/disable navigation based on page count
        doc = self.documents[self.current_document_idx]
        self.prev_btn.setEnabled(doc.current_page > 0)
        self.next_btn.setEnabled(doc.current_page < doc.get_page_count() - 1)
        self.ocr_btn.setEnabled(True)
        
        self.predict_btn.setEnabled(True)

    def update_document_info(self):
        """Update the document information display"""
        if self.current_document_idx >= 0:
            doc = self.documents[self.current_document_idx]
            self.doc_info_label.setText(
                f"Document: {doc.filename}\nPages: {doc.get_page_count()}\nBoxes: {doc.get_box_count()}"
            )
            self.page_label.setText(
                f"Page: {doc.current_page + 1}/{doc.get_page_count()}"
            )
        else:
            self.doc_info_label.setText("No document selected")
            self.page_label.setText("Page: 0/0")

    def run_ocr(self):
        """Run OCR on the current page of the current document"""
        if self.current_document_idx < 0:
            return

        doc = self.documents[self.current_document_idx]
        self.status_label.setText(f"Running OCR on page {doc.current_page + 1}...")

        # Run OCR on the current page
        if doc.run_ocr_on_page(doc.current_page):
            # Clear selected boxes
            self.selected_indices.clear()

            # Update the display
            self.display_current_page()

            # Update document info
            self.update_document_info()

            self.status_label.setText(f"OCR completed for page {doc.current_page + 1}")
        else:
            self.status_label.setText(f"OCR failed for page {doc.current_page + 1}")

    def display_current_page(self):
        """Display the current page with boxes"""
        if self.current_document_idx < 0:
            return

        doc = self.documents[self.current_document_idx]

        # Get the base image
        img_np = doc.get_current_page_image()
        if img_np is None:
            return

        # Create a copy to draw on
        img_copy = img_np.copy()

        # Make sure page_boxes is initialized for the current page
        while len(doc.page_boxes) <= doc.current_page:
            doc.page_boxes.append([])

        # Get the current page's boxes
        self.text_boxes = doc.page_boxes[doc.current_page]

        # Update button states
        self.update_button_states()

        # Draw all boxes
        for idx, box in enumerate(self.text_boxes):
            # Draw rectangle using OpenCV - green for normal, blue for selected
            if idx in self.selected_indices:
                # Selected box color (blue)
                box.draw(img_copy, color=(255, 0, 0))
            else:
                # Normal box color (green) or yellow for labels other than 'O'
                if box.label == "O":
                    color = (0, 255, 0)  # Default green
                else:
                    color = (0, 255, 255)  # Yellow for all non-default labels

                box.draw(img_copy, color=color)

        # Convert the numpy array with annotations to QPixmap
        rgb_image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

        # Properly convert numpy array to QPixmap
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_img)

        # Set pixmap to label
        self.image_label.setPixmap(pixmap)
        self.image_label.setMinimumSize(width, height)

    def prev_page(self):
        """Go to the previous page"""
        if self.current_document_idx < 0:
            return

        doc = self.documents[self.current_document_idx]
        if doc.current_page > 0:
            doc.current_page -= 1
            self.page_label.setText(
                f"Page: {doc.current_page + 1}/{doc.get_page_count()}"
            )

            # Clear selected boxes
            self.selected_indices.clear()

            # Display new page
            self.display_current_page()

            # Update navigation buttons
            self.prev_btn.setEnabled(doc.current_page > 0)
            self.next_btn.setEnabled(doc.current_page < doc.get_page_count() - 1)

    def next_page(self):
        """Go to the next page"""
        if self.current_document_idx < 0:
            return

        doc = self.documents[self.current_document_idx]
        if doc.current_page < doc.get_page_count() - 1:
            doc.current_page += 1
            self.page_label.setText(
                f"Page: {doc.current_page + 1}/{doc.get_page_count()}"
            )

            # Clear selected boxes
            self.selected_indices.clear()

            # Display new page
            self.display_current_page()

            # Update navigation buttons
            self.prev_btn.setEnabled(doc.current_page > 0)
            self.next_btn.setEnabled(doc.current_page < doc.get_page_count() - 1)

    def update_button_states(self):
        """Update button states based on selection status"""
        num_selected = len(self.selected_indices)

        # Enable/disable combine and clear buttons
        has_selection = num_selected > 0

        # Only enable combine if multiple boxes are selected AND they all have the same label
        can_combine = num_selected >= 2
        if can_combine:
            # Check if all selected boxes have the same label
            selected_boxes = [self.text_boxes[idx] for idx in self.selected_indices]
            unique_labels = {box.label for box in selected_boxes}
            can_combine = len(unique_labels) == 1  # Only one unique label

        self.combine_btn.setEnabled(can_combine)
        self.clear_sel_btn.setEnabled(has_selection)

        # Enable/disable label selector
        self.label_selector.setEnabled(num_selected == 1)

        # If exactly one box is selected, update label selector to show its label
        if num_selected == 1:
            box = self.text_boxes[self.selected_indices[0]]
            current_label = box.label
            label_index = (
                self.available_labels.index(current_label)
                if current_label in self.available_labels
                else 0
            )
            self.label_selector.setCurrentIndex(label_index)

    def on_image_click(self, event):
        """Handle mouse clicks on the image for box selection"""
        if not self.text_boxes:
            return

        # Get mouse position relative to the label
        pos = event.pos()

        # Adjust for image scaling if necessary
        pixmap = self.image_label.pixmap()
        if pixmap:
            label_size = self.image_label.size()
            pixmap_size = pixmap.size()

            # Calculate position within the actual image
            scale_x = (
                pixmap_size.width() / label_size.width()
                if label_size.width() > 0
                else 1
            )
            scale_y = (
                pixmap_size.height() / label_size.height()
                if label_size.height() > 0
                else 1
            )

            # If the image is smaller than the label, it's centered
            if pixmap_size.width() < label_size.width():
                offset_x = (label_size.width() - pixmap_size.width()) / 2
                pos.setX(pos.x() - offset_x)

            if pixmap_size.height() < label_size.height():
                offset_y = (label_size.height() - pixmap_size.height()) / 2
                pos.setY(pos.y() - offset_y)

        # Check if the click is on any text box
        for idx, box in enumerate(self.text_boxes):
            if box.contains_point(pos.x(), pos.y()):
                # Toggle selection status
                if idx in self.selected_indices:
                    # Remove the index from the selection list
                    self.selected_indices.remove(idx)
                    self.status_label.setText(f'Box deselected: "{box.text}"')
                else:
                    # Check if adding this box would create a multi-label selection
                    if self.selected_indices:
                        current_label = self.text_boxes[self.selected_indices[0]].label
                        if box.label != current_label:
                            self.status_label.setText(
                                f"Warning: Cannot combine boxes with different labels"
                            )

                    # Add the index to the selection list if not already present
                    if idx not in self.selected_indices:
                        self.selected_indices.append(idx)
                    self.status_label.setText(f'Box selected: "{box.text}"')

                # Update button states
                self.update_button_states()

                # Redraw with updated selections
                self.display_current_page()
                return

    def on_image_hover(self, event):
        """Handle mouse hover events over the image"""
        if not self.text_boxes:
            return

        # Get mouse position relative to the label
        pos = event.pos()

        # Adjust for image scaling if necessary
        pixmap = self.image_label.pixmap()
        if pixmap:
            label_size = self.image_label.size()
            pixmap_size = pixmap.size()

            scale_x = (
                pixmap_size.width() / label_size.width()
                if label_size.width() > 0
                else 1
            )
            scale_y = (
                pixmap_size.height() / label_size.height()
                if label_size.height() > 0
                else 1
            )

            # If the image is smaller than the label, it's centered
            if pixmap_size.width() < label_size.width():
                offset_x = (label_size.width() - pixmap_size.width()) / 2
                pos.setX(pos.x() - offset_x)

            if pixmap_size.height() < label_size.height():
                offset_y = (label_size.height() - pixmap_size.height()) / 2
                pos.setY(pos.y() - offset_y)

        # Check if the mouse is over any text box
        for box_idx, box in enumerate(self.text_boxes):
            if box.contains_point(pos.x(), pos.y()):
                # Store current position and text for the tooltip
                self.hover_position = QCursor.pos()

                # Include the label in the tooltip if it's not the default
                tooltip_text = box.text
                if box.label != "O":
                    tooltip_text = f"[{box.label}] {tooltip_text}"

                self.hover_text = tooltip_text

                # Start the timer for delayed tooltip (500ms)
                self.tooltip_timer.start(500)
                return

        # If we're not over any text box, stop the timer and hide the tooltip
        self.tooltip_timer.stop()
        QToolTip.hideText()

    def show_text_tooltip(self):
        """Show the tooltip with extracted text after the delay"""
        if self.hover_position and self.hover_text:
            QToolTip.showText(
                self.hover_position, f"OCR Text: {self.hover_text}", self.image_label
            )

    def combine_selected_boxes(self):
        """Combine all selected boxes into one larger box"""
        if self.current_document_idx < 0 or len(self.selected_indices) < 2:
            return

        # Get the selected box objects in their selection order
        selected_boxes = [self.text_boxes[idx] for idx in self.selected_indices]

        # Verify all boxes have the same label
        labels = {box.label for box in selected_boxes}
        if len(labels) > 1:
            self.status_label.setText("Cannot combine boxes with different labels")
            return

        # Create a new combined box
        combined_box = self.combine_boxes_in_order(selected_boxes)

        # Create a new list of boxes excluding the selected ones
        doc = self.documents[self.current_document_idx]
        new_text_boxes = [
            box
            for idx, box in enumerate(self.text_boxes)
            if idx not in self.selected_indices
        ]

        # Add the new combined box
        new_text_boxes.append(combined_box)

        # Update the boxes list for the current page
        doc.page_boxes[doc.current_page] = new_text_boxes
        self.text_boxes = new_text_boxes

        # Clear the selection
        self.selected_indices.clear()

        # Update the display
        self.status_label.setText(
            f'Combined {len(selected_boxes)} boxes with label "{combined_box.label}"'
        )
        self.display_current_page()

        # Update document info as box count has changed
        self.update_document_info()

    def combine_boxes_in_order(self, boxes):
        """Create a new box by combining boxes in their current order"""
        if not boxes:
            return None

        # Find the bounding rectangle
        min_x = min(box.x for box in boxes)
        min_y = min(box.y for box in boxes)
        max_x = max(box.x + box.w for box in boxes)
        max_y = max(box.y + box.h for box in boxes)

        # Calculate width and height
        width = max_x - min_x
        height = max_y - min_y

        # Combine all words in the selection order
        all_words = []
        for box in boxes:
            all_words.extend(box.words)

        # All boxes should have the same label at this point
        label = boxes[0].label

        # Create a new box with the same label
        return TextBox(min_x, min_y, width, height, all_words, label)

    def clear_selection(self):
        """Clear all selected boxes"""
        self.selected_indices.clear()
        self.status_label.setText("Selection cleared")
        self.display_current_page()

    def add_new_label(self):
        """Add a new label to the available labels list"""
        label, ok = QInputDialog.getText(
            self, "Add New Label", "Enter new label:", QLineEdit.Normal, ""
        )

        if ok and label:
            if label not in self.available_labels:
                self.available_labels.append(label)
                self.label_selector.addItem(label)
                self.status_label.setText(f'Added new label: "{label}"')

                # If one box is selected, apply the new label
                if len(self.selected_indices) == 1:
                    self.label_selector.setCurrentIndex(
                        self.available_labels.index(label)
                    )
            else:
                self.status_label.setText(f'Label "{label}" already exists')

    def apply_label_to_selection(self):
        """Apply the selected label to the currently selected box"""
        if len(self.selected_indices) != 1 or self.current_document_idx < 0:
            return

        selected_label = self.label_selector.currentText()
        box_idx = self.selected_indices[0]

        # Update the box's label
        self.text_boxes[box_idx].label = selected_label
        self.status_label.setText(f'Applied label "{selected_label}" to selected box')

        # Update the display to reflect any visual changes based on label
        self.display_current_page()

    def show_model_training(self):
        """Show the model training dialog"""
        if not self.documents:
            self.status_label.setText("No documents loaded. Please upload documents first.")
            return
        
        # Check if any documents have labeled boxes
        has_labels = False
        for doc in self.documents:
            for page_boxes in doc.page_boxes:
                for box in page_boxes:
                    if box.label != "O":
                        has_labels = True
                        break
                if has_labels:
                    break
            if has_labels:
                break
        
        if not has_labels:
            self.status_label.setText("No labeled text boxes found. Please label some boxes first.")
            QMessageBox.warning(self, "No Labels", 
                              "No labeled text boxes found. Please add labels to boxes before training.")
            return
        
        dialog = ModelTrainingDialog(self, self.documents)
        dialog.exec_()
        
        # Update status after dialog closes
        self.status_label.setText("Model training dialog closed")
    
    def show_model_prediction(self):
        """Show the model prediction dialog"""
        if self.current_document_idx < 0:
            self.status_label.setText("No document selected.")
            return
        
        doc = self.documents[self.current_document_idx]
        
        # Check if there are any boxes on the current page
        if not doc.page_boxes[doc.current_page]:
            self.status_label.setText("No text boxes found on current page. Run OCR first.")
            QMessageBox.warning(self, "No Text Boxes", 
                              "No text boxes found on current page. Please run OCR first.")
            return
        
        dialog = ModelPredictionDialog(self, doc, doc.current_page)
        dialog.prediction_done.connect(self.apply_predictions)
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            self.status_label.setText("Predictions applied to current page")
        else:
            self.status_label.setText("Model prediction canceled")
    
    def apply_predictions(self, predictions):
        """
        Apply the predictions from the model to the current page
        
        Args:
            predictions: List of dictionaries with word, box, and label information
        """
        if self.current_document_idx < 0 or not predictions:
            return
        
        doc = self.documents[self.current_document_idx]
        
        # Clear current text boxes on the page
        doc.page_boxes[doc.current_page] = []
        
        # Get image dimensions for denormalization
        image = doc.images[doc.current_page]
        width, height = image.size if hasattr(image, 'size') else (image.width, image.height)
        
        # Group predictions by label (except 'O')
        label_groups = {}
        
        for pred in predictions:
            label = pred["label"]
            if label != "O":
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(pred)
        
        from models.text_box import TextBox
        
        # Create new text boxes from model predictions
        for label, items in label_groups.items():
            for item in items:
                # Denormalize the box coordinates
                norm_box = item["box"]
                x1 = int(norm_box[0] * width / 1000)
                y1 = int(norm_box[1] * height / 1000)
                x2 = int(norm_box[2] * width / 1000)
                y2 = int(norm_box[3] * height / 1000)
                
                # Create a new text box
                text_box = TextBox(
                    x=x1,
                    y=y1,
                    w=x2-x1,
                    h=y2-y1,
                    words=[item["word"]],
                    label=label
                )
                
                # Add the box to the document
                doc.page_boxes[doc.current_page].append(text_box)
        
        # Update the text_boxes reference for the current page
        self.text_boxes = doc.page_boxes[doc.current_page]
        
        # Display the updated page
        self.display_current_page()
        
        # Update document info
        self.update_document_info()
        
        self.status_label.setText(f"Applied {len(doc.page_boxes[doc.current_page])} labeled boxes from model predictions")