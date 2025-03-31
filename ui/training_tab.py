"""
PDF Information Extractor - Training Tab
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

from pdf_info_extractor.data.dataset import PDFDataset
from pdf_info_extractor.models.trainer import PDFExtractorTrainer
from pdf_info_extractor.utils.pdf_utils import load_annotations


class TrainingTab:
    def __init__(self, parent, app):
        """Initialize the training tab"""
        self.parent = parent
        self.app = app

        # Create the main frame
        self.frame = ttk.Frame(parent)

        # State variables
        self.annotations_path = None
        self.output_dir = "./output"
        self.training_thread = None

        # Set up UI components
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components"""
        # Main frame
        train_frame = ttk.Frame(self.frame)
        train_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel (settings)
        settings_frame = ttk.LabelFrame(train_frame, text="Training Settings")
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Model settings
        model_frame = ttk.LabelFrame(settings_frame, text="Model")
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(model_frame, text="Model Name:").pack(anchor=tk.W, padx=5, pady=2)
        self.model_name_var = tk.StringVar(value=self.app.config.model["model_name"])
        ttk.Entry(model_frame, textvariable=self.model_name_var).pack(
            fill=tk.X, padx=5, pady=2
        )

        # Training settings
        training_settings_frame = ttk.LabelFrame(settings_frame, text="Training")
        training_settings_frame.pack(fill=tk.X, padx=5, pady=5)

        # Annotations file
        annot_frame = ttk.Frame(training_settings_frame)
        annot_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(annot_frame, text="Annotations:").pack(side=tk.LEFT, padx=2)
        self.annot_var = tk.StringVar()
        ttk.Entry(annot_frame, textvariable=self.annot_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
        ttk.Button(annot_frame, text="Browse", command=self.browse_annotations).pack(
            side=tk.LEFT, padx=2
        )

        # Output directory
        output_frame = ttk.Frame(training_settings_frame)
        output_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(output_frame, text="Output Dir:").pack(side=tk.LEFT, padx=2)
        self.output_dir_var = tk.StringVar(value=self.output_dir)
        ttk.Entry(output_frame, textvariable=self.output_dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2
        )
        ttk.Button(output_frame, text="Browse", command=self.browse_output_dir).pack(
            side=tk.LEFT, padx=2
        )

        # Cross-validation
        cv_frame = ttk.Frame(training_settings_frame)
        cv_frame.pack(fill=tk.X, padx=5, pady=2)
        self.cv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            cv_frame, text="Use Cross-Validation", variable=self.cv_var
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(cv_frame, text="Folds:").pack(side=tk.LEFT, padx=(10, 2))
        self.folds_var = tk.StringVar(value="5")
        ttk.Spinbox(
            cv_frame, from_=2, to=10, textvariable=self.folds_var, width=5
        ).pack(side=tk.LEFT)

        # Learning rate
        lr_frame = ttk.Frame(training_settings_frame)
        lr_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT, padx=2)
        self.lr_var = tk.StringVar(value="1e-5")
        ttk.Entry(lr_frame, textvariable=self.lr_var, width=10).pack(
            side=tk.LEFT, padx=2
        )

        # Batch size
        batch_frame = ttk.Frame(training_settings_frame)
        batch_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(batch_frame, text="Batch Size:").pack(side=tk.LEFT, padx=2)
        self.batch_var = tk.StringVar(value="2")
        ttk.Spinbox(
            batch_frame, from_=1, to=8, textvariable=self.batch_var, width=5
        ).pack(side=tk.LEFT, padx=2)

        # Max steps
        steps_frame = ttk.Frame(training_settings_frame)
        steps_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(steps_frame, text="Max Steps:").pack(side=tk.LEFT, padx=2)
        self.steps_var = tk.StringVar(value="1000")
        ttk.Entry(steps_frame, textvariable=self.steps_var, width=10).pack(
            side=tk.LEFT, padx=2
        )

        # Training button
        self.train_button = ttk.Button(
            settings_frame, text="Start Training", command=self.start_training
        )
        self.train_button.pack(fill=tk.X, padx=5, pady=10)

        # Right panel (logs)
        log_frame = ttk.LabelFrame(train_frame, text="Training Logs")
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Log text widget
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollbar for log
        log_scrollbar = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self.log_text.yview
        )
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)

        # Set read-only
        self.log_text.config(state=tk.DISABLED)

    def browse_annotations(self):
        """Browse for annotations file"""
        file_path = filedialog.askopenfilename(
            title="Select Annotations File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if file_path:
            self.annotations_path = file_path
            self.annot_var.set(file_path)

    def browse_output_dir(self):
        """Browse for output directory"""
        dir_path = filedialog.askdirectory(title="Select Output Directory")

        if dir_path:
            self.output_dir = dir_path
            self.output_dir_var.set(dir_path)

    def add_to_log(self, message):
        """Add message to training log"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.frame.update_idletasks()

    def start_training(self):
        """Start the training process"""
        # Validate inputs
        if not self.annotations_path:
            messagebox.showerror("Error", "Please select an annotations file")
            return

        if not os.path.exists(self.annotations_path):
            messagebox.showerror("Error", "Annotations file doesn't exist")
            return

        # Get training parameters
        self.output_dir = self.output_dir_var.get()
        use_cv = self.cv_var.get()
        num_folds = int(self.folds_var.get())
        try:
            learning_rate = float(self.lr_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid learning rate")
            return

        try:
            batch_size = int(self.batch_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid batch size")
            return

        try:
            max_steps = int(self.steps_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid max steps")
            return

        # Update model name in config
        model_name = self.model_name_var.get()
        if model_name:
            self.app.config.update_model_config(model_name=model_name)

        # Update training config
        self.app.config.update_training_config(
            output_dir=self.output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # Update cross-validation config
        if use_cv:
            self.app.config.update_cv_config(num_folds=num_folds)

        # Disable the train button
        self.train_button.config(state=tk.DISABLED)

        # Clear the log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

        # Start training in a separate thread
        self.add_to_log("Starting training process...")
        self.app.update_status("Training started...")

        self.training_thread = threading.Thread(target=self.train_model)
        self.training_thread.daemon = True
        self.training_thread.start()

    def train_model(self):
        """Training process - runs in a separate thread"""
        try:
            self.add_to_log(f"Loading annotations from {self.annotations_path}...")

            # Load annotations
            pdf_data, label_info = load_annotations(self.annotations_path)

            # Create dataset handler
            self.add_to_log("Creating dataset...")
            dataset_handler = PDFDataset(self.app.config, self.app.processor)

            # Set up labels
            if "labels" in label_info:
                dataset_handler.setup_labels(label_info["labels"])
                self.add_to_log(f"Using labels: {', '.join(label_info['labels'])}")

            if "label2id" in label_info:
                dataset_handler.label2id = label_info["label2id"]

            if "id2label" in label_info:
                dataset_handler.id2label = label_info["id2label"]

            # Create dataset
            self.add_to_log("Creating dataset from annotations...")
            dataset, feature_info = dataset_handler.create_dataset_from_annotations(
                pdf_data
            )
            self.add_to_log(f"Created dataset with {len(dataset)} examples")

            # Process dataset
            self.add_to_log("Processing dataset...")
            processed_dataset = dataset_handler.process_dataset(dataset)

            # Initialize trainer
            self.add_to_log("Initializing trainer...")
            trainer = PDFExtractorTrainer(self.app.config, self.app.processor)

            # Train model
            if self.cv_var.get():
                self.add_to_log(
                    f"Training with {self.app.config.cv['num_folds']}-fold cross-validation..."
                )
                best_model, cv_results = trainer.train_with_cross_validation(
                    processed_dataset, feature_info
                )

                # Log cross-validation results
                self.add_to_log("\nCross-validation results:")
                for i, fold_result in enumerate(cv_results["fold_results"]):
                    self.add_to_log(f"Fold {i+1}: {fold_result}")

                self.add_to_log(f"\nAverage results: {cv_results['average_results']}")
                self.add_to_log(f"Best model from fold {cv_results['best_fold']+1}")

                self.add_to_log(
                    f"\nBest model saved to {os.path.join(self.output_dir, 'best_model')}"
                )
            else:
                self.add_to_log("Training single model...")
                model = trainer.train_single_model(processed_dataset, feature_info)
                self.add_to_log(
                    f"Model saved to {os.path.join(self.output_dir, 'model')}"
                )

            self.add_to_log("\nTraining completed successfully!")
            self.app.update_status("Training completed")

            # Show success message
            messagebox.showinfo(
                "Training Complete", "Model training completed successfully!"
            )
        except Exception as e:
            self.add_to_log(f"Error during training: {str(e)}")
            self.app.update_status("Training failed")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            import traceback

            self.add_to_log(traceback.format_exc())
        finally:
            # Re-enable the train button
            self.train_button.config(state=tk.NORMAL)
