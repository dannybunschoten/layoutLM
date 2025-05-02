from PyQt5.QtWidgets import (
  QDialog,
  QVBoxLayout,
  QHBoxLayout,
  QLabel,
  QPushButton,
  QComboBox,
  QFileDialog,
  QMessageBox,
  QListWidget,
  QGroupBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
import os
from models.train_model import ModelTrainer


class PredictionWorker(QThread):
  """Worker thread for running predictions to keep UI responsive"""

  progress_update = pyqtSignal(str)
  finished_signal = pyqtSignal(bool, str, list)

  def __init__(self, trainer, document, page_idx):
    super().__init__()
    self.trainer = trainer
    self.document = document
    self.page_idx = page_idx

  def run(self):
    try:
      self.progress_update.emit("Running model predictions...")
      predictions = self.trainer.predict(self.document, self.page_idx)
      self.progress_update.emit("Predictions completed!")
      self.finished_signal.emit(
        True, "Predictions completed successfully!", predictions
      )
    except Exception as e:
      self.progress_update.emit(f"Error during prediction: {str(e)}")
      self.finished_signal.emit(False, f"Prediction failed: {str(e)}", [])


class ModelPredictionDialog(QDialog):
  """Dialog for running model predictions on documents"""

  prediction_done = pyqtSignal(list)

  def __init__(self, parent=None, document=None, page_idx=0):
    super().__init__(parent)
    self.document = document
    self.page_idx = page_idx
    self.trainer = ModelTrainer()
    self.worker = None

    self.initUI()

  def initUI(self):
    self.setWindowTitle("Run Model Predictions")
    self.setMinimumWidth(500)

    main_layout = QVBoxLayout(self)

    # Model selection
    model_group = QGroupBox("Select Model")
    model_layout = QVBoxLayout(model_group)

    self.model_path_label = QLabel("Model Path: Not selected")
    model_layout.addWidget(self.model_path_label)

    model_btn_layout = QHBoxLayout()
    self.select_model_btn = QPushButton("Select Model...")
    self.select_model_btn.clicked.connect(self.select_model)
    model_btn_layout.addWidget(self.select_model_btn)

    self.model_options_combo = QComboBox()
    self.update_model_options()
    model_btn_layout.addWidget(self.model_options_combo)

    model_layout.addLayout(model_btn_layout)

    # List available labels from model
    self.labels_list = QListWidget()
    model_layout.addWidget(QLabel("Available Labels:"))
    model_layout.addWidget(self.labels_list)

    main_layout.addWidget(model_group)

    # Document information
    if self.document:
      doc_info = QLabel(
        f"Document: {self.document.filename}, Page: {self.page_idx + 1}"
      )
      main_layout.addWidget(doc_info)

    # Status message
    self.status_label = QLabel("Select a model to begin")
    main_layout.addWidget(self.status_label)

    # Buttons
    button_layout = QHBoxLayout()

    self.predict_btn = QPushButton("Run Prediction")
    self.predict_btn.clicked.connect(self.run_prediction)
    self.predict_btn.setEnabled(False)

    self.cancel_btn = QPushButton("Cancel")
    self.cancel_btn.clicked.connect(self.reject)

    button_layout.addWidget(self.predict_btn)
    button_layout.addWidget(self.cancel_btn)

    main_layout.addLayout(button_layout)

  def update_model_options(self):
    """Update the model options combobox with available models"""
    self.model_options_combo.clear()

    # Add built-in models
    self.model_options_combo.addItem("microsoft/layoutlmv3-base")
    self.model_options_combo.addItem("nielsr/funsd-layoutlmv3")

    # Look for local trained models
    if os.path.exists("trained_model"):
      trained_models = [
        f
        for f in os.listdir("trained_model")
        if os.path.isdir(os.path.join("trained_model", f))
      ]
      for model in trained_models:
        self.model_options_combo.addItem(os.path.join("trained_model", model))

  def select_model(self):
    """Select a model directory"""
    model_dir = QFileDialog.getExistingDirectory(self, "Select Model Directory")
    if model_dir:
      # Check if it looks like a Hugging Face model
      if os.path.exists(os.path.join(model_dir, "config.json")):
        self.model_path_label.setText(f"Model Path: {model_dir}")
        self.model_options_combo.addItem(model_dir)
        self.model_options_combo.setCurrentText(model_dir)
        self.load_model(model_dir)
      else:
        QMessageBox.warning(
          self,
          "Invalid Model Directory",
          "The selected directory does not appear to contain a valid model.",
        )

  def load_model(self, model_path):
    """Load the selected model and update UI"""
    try:
      self.status_label.setText(f"Loading model from {model_path}...")
      self.trainer.load_model(model_path)

      # Update labels list
      self.labels_list.clear()
      if self.trainer.id2label:
        for idx, label in self.trainer.id2label.items():
          self.labels_list.addItem(label)

      self.predict_btn.setEnabled(True)
      self.status_label.setText("Model loaded successfully. Ready to run predictions.")
    except Exception as e:
      self.status_label.setText(f"Error loading model: {str(e)}")
      QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")

  def run_prediction(self):
    """Run predictions on the current document page"""
    if not self.document:
      QMessageBox.warning(
        self, "No Document", "No document is selected for prediction."
      )
      return

    # Get selected model
    model_path = self.model_options_combo.currentText()

    # Load model if not already loaded
    if self.trainer.model is None:
      try:
        self.load_model(model_path)
      except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
        return

    # Update UI
    self.predict_btn.setEnabled(False)
    self.status_label.setText("Running predictions...")

    # Create worker thread
    self.worker = PredictionWorker(
      trainer=self.trainer, document=self.document, page_idx=self.page_idx
    )

    # Connect signals
    self.worker.progress_update.connect(self.update_progress)
    self.worker.finished_signal.connect(self.prediction_finished)

    # Start prediction
    self.worker.start()

  def update_progress(self, message):
    """Update progress display"""
    self.status_label.setText(message)

  def prediction_finished(self, success, message, predictions):
    """Handle prediction completion"""
    self.predict_btn.setEnabled(True)

    if success:
      self.status_label.setText(message)

      # Emit signal with predictions
      self.prediction_done.emit(predictions)

      # Close dialog
      self.accept()
    else:
      self.status_label.setText(f"Error: {message}")
      QMessageBox.critical(self, "Prediction Error", message)
