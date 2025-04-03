import os
import numpy as np
import torch
import time
import json
import copy
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    TrainerCallback
)
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


class MetricsCallback(TrainerCallback):
    """Callback to collect metrics during training"""
    def __init__(self):
        self.metrics_history = []
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Make a deep copy to avoid reference issues
        if metrics:
            self.metrics_history.append(copy.deepcopy(metrics))
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # If we're not evaluating every epoch, we still want to track train loss
        if not self.metrics_history or len(self.metrics_history) < state.epoch:
            metrics = {"train_loss": state.log_history[-1].get("loss", 0)}
            self.metrics_history.append(metrics)


class ModelTrainer:
    """Class to handle the training of document models based on labeled data using built-in OCR"""

    def __init__(self, output_dir="trained_model"):
        """
        Initialize the model trainer
        
        Args:
            output_dir: Directory to save the trained model
        """
        self.output_dir = output_dir
        self.processor = None
        self.model = None
        self.id2label = None
        self.label2id = None
        self.features = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def prepare_training_data(self, documents):
        """
        Prepare training data from labeled documents using built-in OCR
        
        Args:
            documents: List of Document objects with labeled text boxes
            
        Returns:
            Dataset ready for training
        """
        print("\n=== DEBUGGING TRAINING DATA PREPARATION ===")
        train_examples = []
        
        for doc_idx, doc in enumerate(documents):
            print(f"\nDocument {doc_idx}: {doc.filename}")
            
            for page_idx, boxes in enumerate(doc.page_boxes):
                if not boxes:  # Skip pages with no boxes
                    continue
                
                print(f"\n  Page {page_idx} - {len(boxes)} boxes")
                
                # Get the page image
                image = doc.images[page_idx]
                
                # Create a ground truth label dictionary for this page
                # We'll use string representations of box coordinates as keys
                label_dict = {}
                
                for box in boxes:
                    if box.label != "O":
                        # Store box coordinates and label
                        # Convert tuple to string for dictionary key
                        box_key = f"{box.x},{box.y},{box.x + box.w},{box.y + box.h}"
                        label_dict[box_key] = box.label
                
                print(f"  Created label dictionary with {len(label_dict)} labeled boxes")
                
                # Store the image and label dictionary
                train_examples.append({
                    "id": f"{doc.filename}_page{page_idx}",
                    "image": image,
                    "label_dict": label_dict
                })
        
        print(f"\nTotal examples created: {len(train_examples)}")
        print("=== END DEBUGGING ===\n")
        
        # Convert to dataset
        dataset = Dataset.from_list(train_examples)
        return dataset
    
    def prepare_examples(self, examples):
        """
        Process examples for the model using built-in OCR
        
        Args:
            examples: Batch of examples
            
        Returns:
            Processed batch
        """
        # Add detailed debugging
        print(f"prepare_examples called with {len(examples['image'])} examples")
        
        images = examples["image"]
        label_dicts = examples["label_dict"]
        
        # Process each image individually and store results
        all_input_ids = []
        all_attention_masks = []
        all_bbox = []
        all_labels = []
        all_pixel_values = []
        
        for image_idx, (image, label_dict) in enumerate(zip(images, label_dicts)):
            print(f"Processing image {image_idx}")
            
            try:
                # Process single image
                encoding = self.processor(image, return_tensors="pt")
                
                # Debug output
                print(f"  Processor output keys: {list(encoding.keys())}")
                print(f"  Shapes: pixel_values={encoding['pixel_values'].shape}, "
                    f"input_ids={encoding['input_ids'].shape}, "
                    f"attention_mask={encoding['attention_mask'].shape}, "
                    f"bbox={encoding['bbox'].shape}")
                
                # Get individual tensors (remove batch dimension)
                input_ids = encoding["input_ids"][0]
                attention_mask = encoding["attention_mask"][0]
                bbox = encoding["bbox"][0]
                pixel_values = encoding["pixel_values"][0]
                
                # Create labels tensor
                labels = torch.ones_like(input_ids) * -100  # Default: ignore for loss calculation
                
                # Match boxes to our labeled areas
                for i, (box, word_id) in enumerate(zip(bbox, input_ids)):
                    # Skip special tokens
                    if word_id in self.processor.tokenizer.all_special_ids:
                        continue
                    
                    # Get coordinates of this token
                    x_min, y_min, x_max, y_max = box.tolist()
                    
                    # Try to match this box to one of our labeled boxes
                    matched_label = "O"  # Default label
                    
                    for box_key, label in label_dict.items():
                        # Parse the coordinates from the string key
                        box_coords = box_key.split(',')
                        box_x1, box_y1, box_x2, box_y2 = map(int, box_coords)
                        
                        # Check for overlap
                        if (x_min <= box_x2 and x_max >= box_x1 and
                            y_min <= box_y2 and y_max >= box_y1):
                            matched_label = label
                            break
                    
                    # Convert label to id
                    if matched_label in self.label2id:
                        labels[i] = self.label2id[matched_label]
                    else:
                        labels[i] = self.label2id["O"]
                
                # Store prepared tensors
                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
                all_bbox.append(bbox)
                all_labels.append(labels)
                all_pixel_values.append(pixel_values)
                
                print(f"  Successfully processed image {image_idx}, sequence length: {len(input_ids)}")
                
            except Exception as e:
                print(f"Error processing image {image_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
        
        # Debug lengths
        print(f"Number of processed examples: {len(all_input_ids)}")
        for i, ids in enumerate(all_input_ids):
            print(f"Example {i} lengths: input_ids={len(all_input_ids[i])}, "
                f"attention_mask={len(all_attention_masks[i])}, "
                f"bbox={len(all_bbox[i])}, labels={len(all_labels[i])}")
        
        # Simple approach - process one image at a time
        if len(all_input_ids) == 0:
            print("No examples were processed successfully!")
            # Return empty batch
            return {
                "pixel_values": torch.zeros((0, 3, 224, 224)),
                "input_ids": torch.zeros((0, 0), dtype=torch.long),
                "attention_mask": torch.zeros((0, 0), dtype=torch.long),
                "bbox": torch.zeros((0, 0, 4), dtype=torch.long),
                "labels": torch.zeros((0, 0), dtype=torch.long)
            }
        
        if len(all_input_ids) == 1:
            # Only one example - no need for padding
            return {
                "pixel_values": all_pixel_values[0].unsqueeze(0),
                "input_ids": all_input_ids[0].unsqueeze(0),
                "attention_mask": all_attention_masks[0].unsqueeze(0),
                "bbox": all_bbox[0].unsqueeze(0),
                "labels": all_labels[0].unsqueeze(0)
            }
        
        # Pad sequences to the same length for batching
        max_length = max(len(ids) for ids in all_input_ids)
        print(f"Max sequence length: {max_length}")
        
        # Pad all sequences
        padded_input_ids = []
        padded_attention_masks = []
        padded_bbox = []
        padded_labels = []
        
        for i in range(len(all_input_ids)):
            padded_input_ids.append(self._pad_sequence(all_input_ids[i], max_length, self.processor.tokenizer.pad_token_id))
            padded_attention_masks.append(self._pad_sequence(all_attention_masks[i], max_length, 0))
            padded_bbox.append(self._pad_bbox(all_bbox[i], max_length))
            padded_labels.append(self._pad_sequence(all_labels[i], max_length, -100))
        
        # Create batch dictionary
        return {
            "pixel_values": torch.stack(all_pixel_values),
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "bbox": torch.stack(padded_bbox),
            "labels": torch.stack(padded_labels)
        }

    def _pad_sequence(self, sequence, length, pad_value):
        """Pad a 1D sequence to the specified length"""
        current_length = len(sequence)
        if current_length >= length:
            return sequence[:length]
        
        padding = torch.full((length - current_length,), pad_value, dtype=sequence.dtype)
        return torch.cat([sequence, padding])

    def _pad_bbox(self, bbox, length):
        """Pad a 2D bbox tensor to the specified length"""
        current_length = len(bbox)
        if current_length >= length:
            return bbox[:length]
        
        padding_shape = (length - current_length, 4)
        padding = torch.zeros(padding_shape, dtype=bbox.dtype)
        return torch.cat([bbox, padding])

    def _pad_tensor(self, tensor, target_length, padding_value):
        """
        Pad a tensor to the target length
        
        Args:
            tensor: Tensor to pad
            target_length: Desired length
            padding_value: Value to use for padding
            
        Returns:
            Padded tensor
        """
        current_length = len(tensor)
        if current_length >= target_length:
            return tensor[:target_length]
        
        # Create padding
        if isinstance(padding_value, torch.Tensor) and padding_value.dim() > 0:
            # For multi-dimensional values like bbox
            padding = torch.stack([padding_value] * (target_length - current_length))
            return torch.cat([tensor, padding], dim=0)
        else:
            # For scalar values
            padding = torch.ones(target_length - current_length, 
                            dtype=tensor.dtype) * padding_value
            return torch.cat([tensor, padding], dim=0)
    
    def prepare_label_mappings(self, dataset):
        """
        Create id2label and label2id mappings from the dataset
        
        Args:
            dataset: Dataset with label_dict
            
        Returns:
            id2label, label2id dictionaries
        """
        # Get unique labels from all label dictionaries
        unique_labels = set(["O"])  # Always include "O"
        
        for example in dataset:
            unique_labels.update(example["label_dict"].values())
        
        print("\n=== DEBUGGING LABEL MAPPINGS ===")
        print(f"All unique labels: {sorted(unique_labels)}")
        
        # Sort labels (keep "O" first)
        label_list = ["O"] + sorted([l for l in unique_labels if l != "O"])
        
        print(f"Ordered label list: {label_list}")
        
        # Create mappings
        id2label = {i: label for i, label in enumerate(label_list)}
        label2id = {label: i for i, label in enumerate(label_list)}
        
        print("id2label mapping:")
        for id, label in id2label.items():
            print(f"  {id}: '{label}'")
        
        print("label2id mapping:")
        for label, id in label2id.items():
            print(f"  '{label}': {id}")
        
        print("=== END DEBUGGING ===\n")
        
        self.id2label = id2label
        self.label2id = label2id
        
        return id2label, label2id
    
    def compute_metrics(self, p):
        """
        Compute metrics for evaluation
        
        Args:
            p: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
        
        return results
    
    def train(self, train_data, val_data=None, model_name="microsoft/layoutlmv3-base", 
              batch_size=2, learning_rate=5e-5, num_train_epochs=3):
        """
        Train the model on the prepared data
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset (optional)
            model_name: Name of the pre-trained model to use
            batch_size: Batch size for training
            learning_rate: Learning rate
            num_train_epochs: Number of training epochs
            
        Returns:
            Trained model and results dictionary
        """
        # Start timing training
        training_start_time = time.time()
        
        # Prepare label mappings
        self.prepare_label_mappings(train_data)
        
        # Initialize processor with OCR enabled
        self.processor = AutoProcessor.from_pretrained(model_name, apply_ocr=True)
        
        # Define features for set_format to work properly
        self.features = Features({
            'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': Sequence(feature=Value(dtype='int64')),
        })
        
        # Process datasets
        train_dataset = train_data.map(
            self.prepare_examples,
            batched=True,
            remove_columns=train_data.column_names,
        )
        
        if val_data:
            eval_dataset = val_data.map(
                self.prepare_examples,
                batched=True,
                remove_columns=val_data.column_names,
            )
        else:
            eval_dataset = None
        
        # Set the format to PyTorch tensors
        train_dataset.set_format("torch")
        if eval_dataset:
            eval_dataset.set_format("torch")
        
        # Load model
        num_labels = len(self.id2label)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Prepare for collecting metrics history
        metrics_callback = MetricsCallback()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            save_total_limit=2,
            report_to="none",  # Disable wandb, tensorboard, etc.
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor,
            data_collator=default_data_collator,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            callbacks=[metrics_callback],
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Get metrics
        train_metrics = train_result.metrics
        
        # Collect evaluation metrics
        final_metrics = {}
        if eval_dataset:
            eval_metrics = trainer.evaluate()
            final_metrics = {k.replace("eval_", ""): v for k, v in eval_metrics.items()}
        
        # Calculate training time
        training_time = time.time() - training_start_time
        
        # Save the model
        trainer.save_model(self.output_dir)
        
        # Save processor for inference
        self.processor.save_pretrained(self.output_dir)
        
        # Create results dictionary
        results = self.create_results_dict(
            model_name=model_name,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_train_epochs,
            training_time=training_time,
            final_metrics=final_metrics,
            history_metrics=metrics_callback.metrics_history,
            label_list=list(self.id2label.values())
        )
        
        # Save the results
        self.save_results(results)
        
        return self.model, results
    
    def predict(self, document, page_idx):
        """
        Predict labels for a page in the document using built-in OCR
        
        Args:
            document: Document object
            page_idx: Page index to predict
            
        Returns:
            Dictionary with predictions and OCR results
        """
        if not self.model or not self.processor:
            raise ValueError("Model or processor not initialized. Run train first.")
        
        # Get image from the document
        image = document.images[page_idx]
        
        # Process the image with the processor's OCR
        encoding = self.processor(image, return_tensors="pt")
        
        # Extract original word boxes and words for later reference
        word_boxes = encoding.pop("bbox", None)[0]
        word_ids = encoding.pop("input_ids", None)[0]
        
        # Get predictions from the model
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Get logits and predictions
        logits = outputs.logits
        predictions = logits.argmax(-1).squeeze().tolist()
        
        # Convert to list if it's a single prediction
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        # Map predictions back to words and boxes
        result = []
        
        # Decode tokens to words for display
        tokenizer = self.processor.tokenizer
        
        for i, (word_id, box, pred) in enumerate(zip(word_ids, word_boxes, predictions)):
            # Skip special tokens
            if word_id in tokenizer.all_special_ids:
                continue
                
            word = tokenizer.decode([word_id])
            if pred != -100:
                label = self.id2label[pred]
            else:
                label = "O"
                
            # Convert normalized box back to pixel coordinates
            x1, y1, x2, y2 = box.tolist()
            
            # Store the prediction
            result.append({
                "word": word,
                "box": [x1, y1, x2, y2],  # Keep normalized for consistency
                "label": label
            })
        
        print(f"Generated {len(result)} predictions")
        return result
    
    def create_results_dict(self, model_name, batch_size, learning_rate, num_epochs, 
                        training_time, final_metrics, history_metrics, label_list):
        """Create a dictionary of training results for visualization"""
        results = {
            "model_info": {
                "model_name": model_name,
                "training_time": training_time,
                "final_f1": final_metrics.get("f1", 0),
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs
            }
        }
        
        # Add learning curve data
        if history_metrics:
            epochs = list(range(1, len(history_metrics) + 1))
            
            # Extract metrics from history
            train_loss = [metrics.get("train_loss", 0) for metrics in history_metrics]
            eval_loss = [metrics.get("eval_loss", 0) for metrics in history_metrics]
            train_f1 = [metrics.get("train_f1", 0) for metrics in history_metrics]
            eval_f1 = [metrics.get("eval_f1", 0) for metrics in history_metrics]
            
            results["learning_curve"] = {
                "epochs": epochs,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "train_f1": train_f1,
                "eval_f1": eval_f1
            }
        
        # Add per-class metrics if available
        if final_metrics:
            # Check if we have per-class metrics
            precision_keys = [k for k in final_metrics.keys() if k.startswith("precision_")]
            recall_keys = [k for k in final_metrics.keys() if k.startswith("recall_")]
            f1_keys = [k for k in final_metrics.keys() if k.startswith("f1_")]
            
            # If we have per-class metrics, add them
            if precision_keys and recall_keys and f1_keys:
                class_metrics = {
                    "precision": {},
                    "recall": {},
                    "f1": {}
                }
                
                # Extract class names from metric keys
                for key in precision_keys:
                    class_name = key.replace("precision_", "")
                    class_metrics["precision"][class_name] = final_metrics.get(key, 0)
                
                for key in recall_keys:
                    class_name = key.replace("recall_", "")
                    class_metrics["recall"][class_name] = final_metrics.get(key, 0)
                
                for key in f1_keys:
                    class_name = key.replace("f1_", "")
                    class_metrics["f1"][class_name] = final_metrics.get(key, 0)
                
                results["class_metrics"] = class_metrics
        
        return results

    def save_results(self, results):
        """Save training results to a JSON file"""
        results_path = os.path.join(self.output_dir, "training_results.json")
        
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            
    def load_model(self, model_dir):
        """
        Load a trained model from a directory
        
        Args:
            model_dir: Directory containing the model
            
        Returns:
            Loaded model
        """
        self.processor = AutoProcessor.from_pretrained(model_dir, apply_ocr=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        
        # Get label mappings from the model config
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        return self.model