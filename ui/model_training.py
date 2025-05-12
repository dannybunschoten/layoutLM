from PyQt5.QtWidgets import (
  QDialog,
  QVBoxLayout,
  QHBoxLayout,
  QLabel,
  QLineEdit,
  QPushButton,
  QComboBox,
  QSpinBox,
  QDoubleSpinBox,
  QProgressBar,
  QCheckBox,
  QListWidget,
  QMessageBox,
  QFileDialog,
  QGroupBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
import torch
from models.train_model import ModelTrainer


class TrainingWorker(QThread):
  """Worker thread for training to keep UI responsive"""

  progress_update = pyqtSignal(str)
  finished_signal = pyqtSignal(bool, str)

  def __init__(
    self, trainer, documents, val_split, model_name, batch_size, learning_rate, epochs
  ):
    super().__init__()
    self.trainer = trainer
    self.documents = documents
    self.val_split = val_split
    self.model_name = model_name
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.epochs = epochs

  def run(self):
    try:
      self.progress_update.emit("Preparing document data...")

      # Check if we have valid documents
      num_boxes = sum(doc.get_box_count() for doc in self.documents)
      self.progress_update.emit(
        f"Processing {len(self.documents)} documents with {num_boxes} text boxes"
      )

      if num_boxes == 0:
        raise ValueError("No text boxes found in the selected documents")

      self.progress_update.emit("Starting training...")
      results = self.trainer.train(
        documents=self.documents,
        val_split=self.val_split,
        model_name=self.model_name,
        batch_size=self.batch_size,
        learning_rate=self.learning_rate,
        epochs=self.epochs,
      )

      # Create a summary message
      runtime = results.get("train_runtime", 0)
      runtime_str = (
        f"{runtime:.1f} seconds" if runtime < 60 else f"{runtime / 60:.1f} minutes"
      )

      summary = f"Training completed in {runtime_str}\n"
      summary += f"Model saved to: {results['model_path']}"

      self.progress_update.emit("Training completed!")
      self.finished_signal.emit(True, summary)

    except Exception as e:
      import traceback

      error_message = f"Training failed: {str(e)}\n\n"
      error_message += traceback.format_exc()
      self.progress_update.emit(f"Error during training: {str(e)}")
      self.finished_signal.emit(False, error_message)


class ModelTrainingDialog(QDialog):
  """Dialog for model training settings and progress"""

  def __init__(self, parent=None, documents=None):
    super().__init__(parent)
    self.documents = documents or []
    self.trainer = ModelTrainer()
    self.worker = None

    self.initUI()

  def initUI(self):
    self.setWindowTitle("Train Document Model")
    self.setMinimumWidth(500)

    main_layout = QVBoxLayout(self)

    # Document selection
    doc_group = QGroupBox("Documents")
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

    # Model settings
    model_group = QGroupBox("Model Settings")
    model_layout = QVBoxLayout(model_group)

    # Model name selection
    model_name_layout = QHBoxLayout()
    model_name_layout.addWidget(QLabel("Base Model:"))
    self.model_name_combo = QComboBox()
    self.model_name_combo.addItems(
      [
        "microsoft/layoutlmv3-base",
        "microsoft/layoutlmv3-large",
        "nielsr/funsd-layoutlmv3",
      ]
    )
    model_name_layout.addWidget(self.model_name_combo)
    model_layout.addLayout(model_name_layout)

    # Output directory
    output_dir_layout = QHBoxLayout()
    output_dir_layout.addWidget(QLabel("Output Directory:"))
    self.output_dir_edit = QLineEdit("trained_model")
    output_dir_layout.addWidget(self.output_dir_edit)
    self.browse_btn = QPushButton("Browse...")
    self.browse_btn.clicked.connect(self.browse_output_dir)
    output_dir_layout.addWidget(self.browse_btn)
    model_layout.addLayout(output_dir_layout)

    # Training parameters
    params_layout = QHBoxLayout()

    # Batch size
    batch_layout = QVBoxLayout()
    batch_layout.addWidget(QLabel("Batch Size:"))
    self.batch_size_spin = QSpinBox()
    self.batch_size_spin.setRange(1, 16)
    self.batch_size_spin.setValue(2)
    batch_layout.addWidget(self.batch_size_spin)
    params_layout.addLayout(batch_layout)

    # Learning rate
    lr_layout = QVBoxLayout()
    lr_layout.addWidget(QLabel("Learning Rate:"))
    self.lr_spin = QDoubleSpinBox()
    self.lr_spin.setDecimals(6)
    self.lr_spin.setRange(0.000001, 0.1)
    self.lr_spin.setSingleStep(0.00001)
    self.lr_spin.setValue(0.00005)
    lr_layout.addWidget(self.lr_spin)
    params_layout.addLayout(lr_layout)

    # Epochs
    epochs_layout = QVBoxLayout()
    epochs_layout.addWidget(QLabel("Epochs:"))
    self.epochs_spin = QSpinBox()
    self.epochs_spin.setRange(1, 100)
    self.epochs_spin.setValue(3)
    epochs_layout.addWidget(self.epochs_spin)
    params_layout.addLayout(epochs_layout)

    model_layout.addLayout(params_layout)

    # Validation split
    val_layout = QHBoxLayout()
    self.use_val_check = QCheckBox("Use validation split (20%)")
    self.use_val_check.setChecked(True)
    val_layout.addWidget(self.use_val_check)
    model_layout.addLayout(val_layout)

    main_layout.addWidget(model_group)

    # Progress tracking
    progress_group = QGroupBox("Training Progress")
    progress_layout = QVBoxLayout(progress_group)

    self.status_label = QLabel("Ready to train")
    progress_layout.addWidget(self.status_label)

    self.progress_bar = QProgressBar()
    self.progress_bar.setRange(0, 0)  # Indeterminate progress
    self.progress_bar.setVisible(False)
    progress_layout.addWidget(self.progress_bar)

    main_layout.addWidget(progress_group)

    # Buttons
    button_layout = QHBoxLayout()
    self.train_btn = QPushButton("Start Training")
    self.train_btn.clicked.connect(self.start_training)

    self.cancel_btn = QPushButton("Cancel")
    self.cancel_btn.clicked.connect(self.reject)

    button_layout.addWidget(self.train_btn)
    button_layout.addWidget(self.cancel_btn)

    main_layout.addLayout(button_layout)

  def update_document_list(self):
    """Update the document list with available documents"""
    self.doc_list.clear()

    for doc in self.documents:
      # Only add documents that have at least one page with labeled boxes
      has_labels = False
      box_count = 0
      for page_boxes in doc.page_boxes:
        for box in page_boxes:
          if box.label != "O":
            has_labels = True
            box_count += 1

      if has_labels:
        item_text = (
          f"{doc.filename} ({doc.get_page_count()} pages, {box_count} labeled boxes)"
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
        self.doc_list.item(self.doc_list.count() - 1).setCheckState(
          Qt.Checked
        )  # Default to checked

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

  def start_training(self):
    """Start the training process"""
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
      msg = "Training on CPU may be slow. Do you want to continue?"
      reply = QMessageBox.question(
        self, "No GPU Detected", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No
      )
      if reply == QMessageBox.No:
        return

    # Get selected documents
    selected_docs = self.get_selected_documents()
    if not selected_docs:
      QMessageBox.warning(
        self,
        "No Documents Selected",
        "Please select at least one document for training.",
      )
      return

    # Check for documents with labels
    labeled_count = 0
    for doc in selected_docs:
      for page_boxes in doc.page_boxes:
        for box in page_boxes:
          if box.label != "O":
            labeled_count += 1

    if labeled_count == 0:
      QMessageBox.warning(
        self,
        "No Labels Found",
        "None of the selected documents have labeled text boxes. Please label some text boxes first.",
      )
      return

    # Update UI
    self.train_btn.setEnabled(False)
    self.cancel_btn.setText("Close")
    self.progress_bar.setVisible(True)
    self.status_label.setText("Preparing to train...")

    # Set up the trainer with output directory
    output_dir = self.output_dir_edit.text()
    self.trainer = ModelTrainer(output_dir=output_dir)

    # Create worker thread
    self.worker = TrainingWorker(
      trainer=self.trainer,
      documents=selected_docs,
      val_split=0.25 if self.use_val_check.isChecked() else 0,
      model_name=self.model_name_combo.currentText(),
      batch_size=self.batch_size_spin.value(),
      learning_rate=self.lr_spin.value(),
      epochs=self.epochs_spin.value(),
    )

    # Connect signals
    self.worker.progress_update.connect(self.update_progress)
    self.worker.finished_signal.connect(self.training_finished)

    # Start training
    self.worker.start()

  def update_progress(self, message):
    """Update progress display"""
    self.status_label.setText(message)

  def training_finished(self, success, message):
    """Handle training completion"""
    self.progress_bar.setVisible(False)
    self.train_btn.setEnabled(True)

    if success:
      QMessageBox.information(self, "Training Complete", message)
    else:
      QMessageBox.critical(self, "Training Error", message)
