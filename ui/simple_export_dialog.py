from PyQt5.QtWidgets import (
  QDialog,
  QVBoxLayout,
  QHBoxLayout,
  QLabel,
  QPushButton,
  QLineEdit,
  QFileDialog,
  QMessageBox,
  QListWidget,
  QProgressBar,
  QGroupBox,
  QCheckBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
import os
from utils.simple_exporter import SimpleExporter


class SimpleExportWorker(QThread):
  """Worker thread for dataset export"""

  progress_update = pyqtSignal(str)
  finished_signal = pyqtSignal(bool, str, str)

  def __init__(self, documents, output_dir, normalize_boxes, include_unlabeled):
    super().__init__()
    self.documents = documents
    self.output_dir = output_dir
    self.normalize_boxes = normalize_boxes
    self.include_unlabeled = include_unlabeled

  def run(self):
    try:
      self.progress_update.emit("Exporting dataset...")
      json_path = SimpleExporter.export_dataset(
        self.documents,
        self.output_dir,
        normalize_boxes=self.normalize_boxes,
        include_unlabeled=self.include_unlabeled,
      )
      script_path = SimpleExporter.create_loader_script(self.output_dir)
      self.progress_update.emit("Dataset export completed!")
      self.finished_signal.emit(True, "Dataset exported successfully!", json_path)
    except Exception as e:
      self.progress_update.emit(f"Error during export: {str(e)}")
      self.finished_signal.emit(False, f"Export failed: {str(e)}", "")


class SimpleExportDialog(QDialog):
  """Dialog for exporting dataset in a simple JSON format"""

  def __init__(self, parent=None, documents=None):
    super().__init__(parent)
    self.documents = documents or []
    self.worker = None

    self.initUI()

  def initUI(self):
    self.setWindowTitle("Export Dataset")
    self.setMinimumWidth(500)

    main_layout = QVBoxLayout(self)

    # Document selection
    doc_group = QGroupBox("Documents to Export")
    doc_layout = QVBoxLayout(doc_group)

    self.doc_list = QListWidget()
    self.update_document_list()
    doc_layout.addWidget(self.doc_list)

    doc_buttons = QHBoxLayout()
    self.select_all_btn = QPushButton("Select All")
    self.select_all_btn.clicked.connect(self.select_all_documents)
    self.deselect_all_btn = QPushButton("Deselect All")
    self.deselect_all_btn.clicked.connect(self.deselect_all_documents)

    doc_buttons.addWidget(self.select_all_btn)
    doc_buttons.addWidget(self.deselect_all_btn)
    doc_layout.addLayout(doc_buttons)

    main_layout.addWidget(doc_group)

    # Export settings
    export_group = QGroupBox("Export Settings")
    export_layout = QVBoxLayout(export_group)

    # Output directory
    output_dir_layout = QHBoxLayout()
    output_dir_layout.addWidget(QLabel("Output Directory:"))
    self.output_dir_edit = QLineEdit(os.path.join(os.getcwd(), "exported_dataset"))
    output_dir_layout.addWidget(self.output_dir_edit)
    self.browse_btn = QPushButton("Browse...")
    self.browse_btn.clicked.connect(self.browse_output_dir)
    output_dir_layout.addWidget(self.browse_btn)
    export_layout.addLayout(output_dir_layout)

    # Options
    options_layout = QVBoxLayout()
    self.normalize_boxes_check = QCheckBox("Normalize bounding boxes to 0-1000 range")
    self.normalize_boxes_check.setChecked(True)
    options_layout.addWidget(self.normalize_boxes_check)

    self.include_unlabeled_check = QCheckBox("Include unlabeled text boxes ('O' label)")
    self.include_unlabeled_check.setChecked(True)
    options_layout.addWidget(self.include_unlabeled_check)

    export_layout.addLayout(options_layout)

    # Export notes and info
    export_layout.addWidget(
      QLabel(
        "This will export your labeled documents in a simple JSON format with:\n"
        "- Document page images\n"
        "- Text boxes with coordinates\n"
        "- A Python script to load and visualize the data"
      )
    )

    main_layout.addWidget(export_group)

    # Progress
    progress_group = QGroupBox("Export Progress")
    progress_layout = QVBoxLayout(progress_group)

    self.status_label = QLabel("Ready to export")
    progress_layout.addWidget(self.status_label)

    self.progress_bar = QProgressBar()
    self.progress_bar.setRange(0, 0)  # Indeterminate progress
    self.progress_bar.setVisible(False)
    progress_layout.addWidget(self.progress_bar)

    main_layout.addWidget(progress_group)

    # Buttons
    button_layout = QHBoxLayout()
    self.export_btn = QPushButton("Export Dataset")
    self.export_btn.clicked.connect(self.export_dataset)

    self.cancel_btn = QPushButton("Cancel")
    self.cancel_btn.clicked.connect(self.reject)

    button_layout.addWidget(self.export_btn)
    button_layout.addWidget(self.cancel_btn)

    main_layout.addLayout(button_layout)

  def update_document_list(self):
    """Update the document list with available documents"""
    self.doc_list.clear()

    for doc in self.documents:
      # Only add documents that have at least one page with labeled boxes
      has_boxes = False
      for page_boxes in doc.page_boxes:
        if page_boxes:
          has_boxes = True
          break

      if has_boxes:
        item_text = (
          f"{doc.filename} ({doc.get_page_count()} pages, {doc.get_box_count()} boxes)"
        )
        self.doc_list.addItem(item_text)
        # Store the document index as item data
        self.doc_list.item(self.doc_list.count() - 1).setData(
          Qt.UserRole, self.documents.index(doc)
        )
        # Set checkable
        self.doc_list.item(self.doc_list.count() - 1).setFlags(
          self.doc_list.item(self.doc_list.count() - 1).flags() | Qt.ItemIsUserCheckable
        )
        self.doc_list.item(self.doc_list.count() - 1).setCheckState(Qt.Checked)

  def select_all_documents(self):
    """Select all documents in the list"""
    for i in range(self.doc_list.count()):
      self.doc_list.item(i).setCheckState(Qt.Checked)

  def deselect_all_documents(self):
    """Deselect all documents in the list"""
    for i in range(self.doc_list.count()):
      self.doc_list.item(i).setCheckState(Qt.Unchecked)

  def browse_output_dir(self):
    """Browse for output directory"""
    dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
    if dir_path:
      self.output_dir_edit.setText(dir_path)

  def get_selected_documents(self):
    """Get list of selected documents"""
    selected_docs = []
    for i in range(self.doc_list.count()):
      if self.doc_list.item(i).checkState() == Qt.Checked:
        doc_idx = self.doc_list.item(i).data(Qt.UserRole)
        selected_docs.append(self.documents[doc_idx])
    return selected_docs

  def export_dataset(self):
    """Start the dataset export process"""
    selected_docs = self.get_selected_documents()

    if not selected_docs:
      QMessageBox.warning(
        self, "No Documents Selected", "Please select at least one document for export."
      )
      return

    output_dir = self.output_dir_edit.text()
    if not output_dir:
      QMessageBox.warning(
        self, "No Output Directory", "Please select an output directory."
      )
      return

    # Confirm if directory exists and has content
    if os.path.exists(output_dir) and os.listdir(output_dir):
      reply = QMessageBox.question(
        self,
        "Directory Not Empty",
        f"The directory '{output_dir}' is not empty. Content may be overwritten. Continue?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
      )
      if reply == QMessageBox.No:
        return

    # Get export options
    normalize_boxes = self.normalize_boxes_check.isChecked()
    include_unlabeled = self.include_unlabeled_check.isChecked()

    # Update UI
    self.export_btn.setEnabled(False)
    self.cancel_btn.setText("Close")
    self.progress_bar.setVisible(True)
    self.status_label.setText("Preparing to export...")

    # Create worker thread
    self.worker = SimpleExportWorker(
      documents=selected_docs,
      output_dir=output_dir,
      normalize_boxes=normalize_boxes,
      include_unlabeled=include_unlabeled,
    )

    # Connect signals
    self.worker.progress_update.connect(self.update_progress)
    self.worker.finished_signal.connect(self.export_finished)

    # Start export
    self.worker.start()

  def update_progress(self, message):
    """Update progress display"""
    self.status_label.setText(message)

  def export_finished(self, success, message, json_path):
    """Handle export completion"""
    self.progress_bar.setVisible(False)

    if success:
      self.status_label.setText(message)

      info_text = (
        f"Dataset exported to: {json_path}\n\n"
        f"The export includes:\n"
        f"- JSON file with all labeled data\n"
        f"- Images for each document page\n"
        f"- A Python script to load and visualize the data\n\n"
        f"You can run the script with:\n"
        f"python load_dataset.py [page_index]"
      )

      QMessageBox.information(self, "Export Complete", info_text)
    else:
      self.status_label.setText(f"Error: {message}")
      QMessageBox.critical(self, "Export Error", message)
