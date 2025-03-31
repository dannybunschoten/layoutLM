"""
Training logic for PDF information extraction models with cross-validation
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import TrainingArguments, Trainer, PreTrainedTokenizer
from transformers import LayoutLMv3ForTokenClassification
from datasets import Dataset
from sklearn.model_selection import KFold
from tqdm import tqdm

from ..config import Config
from ..utils.metrics import compute_metrics


class PDFExtractorTrainer:
    def __init__(self, config: Config, processor, model=None):
        """
        Initialize trainer for PDF information extraction

        Args:
            config: Configuration object
            processor: LayoutLM processor for feature extraction
            model: Pre-trained model (optional)
        """
        self.config = config
        self.processor = processor
        self.model = model
        self.trained_models = []
        self.cv_results = []

    def init_model(
        self, num_labels: int, id2label: Dict[int, str], label2id: Dict[str, int]
    ) -> LayoutLMv3ForTokenClassification:
        """
        Initialize model with proper labels

        Args:
            num_labels: Number of labels
            id2label: Mapping from label IDs to label strings
            label2id: Mapping from label strings to label IDs

        Returns:
            Initialized model
        """
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            self.config.model["model_name"],
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        return model

    def train_with_cross_validation(
        self, dataset: Dataset, feature_info: Dict[str, Any]
    ) -> Tuple[LayoutLMv3ForTokenClassification, Dict[str, Any]]:
        """
        Train model with cross-validation

        Args:
            dataset: Processed dataset
            feature_info: Dictionary with dataset feature information

        Returns:
            tuple: (best model, cross-validation results)
        """
        # Get feature info
        num_labels = feature_info["num_labels"]
        id2label = feature_info["id2label"]
        label2id = feature_info["label2id"]

        # Setup cross-validation
        cv_config = self.config.cv
        kf = KFold(
            n_splits=cv_config["num_folds"],
            shuffle=cv_config["shuffle"],
            random_state=cv_config["random_seed"],
        )

        # Convert dataset to numpy array of indices
        all_indices = np.arange(len(dataset))

        # List to store models and results
        self.trained_models = []
        self.cv_results = []

        print(f"Starting {cv_config['num_folds']}-fold cross-validation")

        for fold, (train_idx, eval_idx) in enumerate(kf.split(all_indices)):
            print(f"Training fold {fold+1}/{cv_config['num_folds']}...")

            # Create train and eval datasets for this fold
            train_dataset_fold = dataset.select(train_idx.tolist())
            eval_dataset_fold = dataset.select(eval_idx.tolist())

            # Initialize a new model for this fold
            fold_model = self.init_model(num_labels, id2label, label2id)

            # Update output directory for this fold
            training_args = TrainingArguments(**self.config.training)
            training_args.output_dir = os.path.join(
                training_args.output_dir, f"fold-{fold+1}"
            )

            # Define trainer for this fold
            trainer = Trainer(
                model=fold_model,
                args=training_args,
                train_dataset=train_dataset_fold,
                eval_dataset=eval_dataset_fold,
                tokenizer=self.processor.tokenizer,
                compute_metrics=compute_metrics,
            )

            # Train the model
            trainer.train()

            # Evaluate on the validation set for this fold
            eval_results = trainer.evaluate()

            # Store the model and its metrics
            self.trained_models.append(fold_model)
            self.cv_results.append(eval_results)

            print(f"Fold {fold+1} results:", eval_results)

        # Compute average metrics across all folds
        avg_results = {}
        for metric in self.cv_results[0].keys():
            avg_results[metric] = sum(fold[metric] for fold in self.cv_results) / len(
                self.cv_results
            )

        print("Average results across all folds:", avg_results)

        # Find the best model based on F1 score
        best_fold_idx = np.argmax([results["eval_f1"] for results in self.cv_results])
        best_model = self.trained_models[best_fold_idx]

        # Save the best model
        best_model_dir = os.path.join(self.config.training["output_dir"], "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        best_model.save_pretrained(best_model_dir)
        self.processor.tokenizer.save_pretrained(best_model_dir)

        # Save label mappings
        with open(os.path.join(best_model_dir, "label_mapping.json"), "w") as f:
            json.dump(
                {
                    "label_list": feature_info["label_list"],
                    "label2id": feature_info["label2id"],
                    "id2label": {
                        str(k): v for k, v in feature_info["id2label"].items()
                    },
                },
                f,
                indent=2,
            )

        # Set the best model as the current model
        self.model = best_model

        # Return the best model and cross-validation results
        return best_model, {
            "fold_results": self.cv_results,
            "average_results": avg_results,
            "best_fold": best_fold_idx,
        }

    def train_single_model(
        self, dataset: Dataset, feature_info: Dict[str, Any]
    ) -> LayoutLMv3ForTokenClassification:
        """
        Train a single model without cross-validation

        Args:
            dataset: Processed dataset
            feature_info: Dictionary with dataset feature information

        Returns:
            Trained model
        """
        # Split the dataset into train and evaluation sets
        dataset = dataset.train_test_split(test_size=0.2)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # Get feature info
        num_labels = feature_info["num_labels"]
        id2label = feature_info["id2label"]
        label2id = feature_info["label2id"]

        # Initialize model
        model = self.init_model(num_labels, id2label, label2id)

        # Define training arguments
        training_args = TrainingArguments(**self.config.training)

        # Define trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor.tokenizer,
            compute_metrics=compute_metrics,
        )

        # Train the model
        trainer.train()

        # Evaluate
        eval_results = trainer.evaluate()
        print("Evaluation results:", eval_results)

        # Save model
        model_dir = os.path.join(self.config.training["output_dir"], "model")
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        self.processor.tokenizer.save_pretrained(model_dir)

        # Save label mappings
        with open(os.path.join(model_dir, "label_mapping.json"), "w") as f:
            json.dump(
                {
                    "label_list": feature_info["label_list"],
                    "label2id": feature_info["label2id"],
                    "id2label": {
                        str(k): v for k, v in feature_info["id2label"].items()
                    },
                },
                f,
                indent=2,
            )

        # Set the model as the current model
        self.model = model

        return model

    def load_model(self, model_path: str) -> LayoutLMv3ForTokenClassification:
        """
        Load model from a directory

        Args:
            model_path: Path to model directory

        Returns:
            Loaded model
        """
        # Load model
        model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)

        # Load label mappings
        label_mapping_path = os.path.join(model_path, "label_mapping.json")
        if os.path.exists(label_mapping_path):
            with open(label_mapping_path, "r") as f:
                label_mapping = json.load(f)
                # TODO: Update processor with label mapping if needed

        self.model = model
        return model
