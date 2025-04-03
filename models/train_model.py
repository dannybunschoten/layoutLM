import os
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height))
    ]

class ModelTrainer:
    """Class to handle the training of document models based on labeled data"""

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
        Prepare training data from labeled documents
        
        Args:
            documents: List of Document objects with labeled text boxes
            
        Returns:
            Dataset ready for training
        """
        train_examples = []
        
        for doc in documents:
            for page_idx, boxes in enumerate(doc.page_boxes):
                if not boxes:  # Skip pages with no boxes
                    continue
                    
                # Get the page image
                image = doc.images[page_idx]
                
                # Get image dimensions for normalization
                width, height = image.size if hasattr(image, 'size') else (image.width, image.height)
                
                # Prepare tokens, bboxes, and labels
                tokens = []
                bboxes = []
                ner_tags = []
                
                for box in boxes:
                    for word in box.words:
                        tokens.append(word)
                        
                        # Normalize bbox coordinates to 0-1000 range
                        normalized_bbox = normalize_box(
                            [box.x, box.y, box.x + box.w, box.y + box.h], 
                            width, 
                            height
                        )
                        bboxes.append(normalized_bbox)
                        
                        # Use the label for this box
                        label = box.label
                        
                        # Determine if this is the first word in the box (B- prefix)
                        # or a continuation (I- prefix)
                        if word == box.words[0] and label != "O":
                            ner_tags.append(f"B-{label}")
                        elif label != "O":
                            ner_tags.append(f"I-{label}")
                        else:
                            ner_tags.append(label)  # "O" remains as is
                
                train_examples.append({
                    "id": f"{doc.filename}_page{page_idx}",
                    "tokens": tokens,
                    "bboxes": bboxes,
                    "ner_tags": ner_tags,
                    "image": image
                })
        
        # Convert to dataset
        dataset = Dataset.from_list(train_examples)
        return dataset
    
    def prepare_label_mappings(self, dataset):
        """
        Create id2label and label2id mappings from the dataset
        
        Args:
            dataset: Dataset with ner_tags
            
        Returns:
            id2label, label2id dictionaries
        """
        # Get unique labels
        unique_labels = set()
        for example in dataset:
            unique_labels.update(example["ner_tags"])
        
        # Sort labels (keep "O" first)
        if "O" in unique_labels:
            unique_labels.remove("O")
            label_list = ["O"] + sorted(unique_labels)
        else:
            label_list = sorted(unique_labels)
        
        # Create mappings
        id2label = {i: label for i, label in enumerate(label_list)}
        label2id = {label: i for i, label in enumerate(label_list)}
        
        self.id2label = id2label
        self.label2id = label2id
        
        return id2label, label2id
    
    def prepare_examples(self, examples):
        """
        Process examples for the model
        
        Args:
            examples: Batch of examples
            
        Returns:
            Processed batch
        """
        images = examples["image"]
        words = examples["tokens"]
        boxes = examples["bboxes"]
        
        # Convert string labels to ids using label2id mapping
        word_labels = []
        for example_labels in examples["ner_tags"]:
            example_label_ids = []
            for label in example_labels:
                if label in self.label2id:
                    example_label_ids.append(self.label2id[label])
                else:
                    example_label_ids.append(self.label2id["O"])  # Default to "O" if not found
            word_labels.append(example_label_ids)
        
        encoding = self.processor(
            images, 
            words, 
            boxes=boxes, 
            word_labels=word_labels,
            truncation=True, 
            padding="max_length"
        )
        
        return encoding
    
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
            Trained model
        """
        # Prepare label mappings
        self.prepare_label_mappings(train_data)
        
        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
        
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
            features=self.features,
        )
        
        if val_data:
            eval_dataset = val_data.map(
                self.prepare_examples,
                batched=True,
                remove_columns=val_data.column_names,
                features=self.features,
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
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model(self.output_dir)
        
        # Save processor for inference
        self.processor.save_pretrained(self.output_dir)
        
        return self.model
    
    def predict(self, document, page_idx):
        """
        Predict labels for a page in the document
        
        Args:
            document: Document object
            page_idx: Page index to predict
            
        Returns:
            List of predicted labels for text boxes
        """
        if not self.model or not self.processor:
            raise ValueError("Model or processor not initialized. Run train first.")
        
        # Get image and boxes from the document
        image = document.images[page_idx]
        boxes = document.page_boxes[page_idx]
        
        if not boxes:
            return []
        
        # Get image dimensions for normalization
        width, height = image.size if hasattr(image, 'size') else (image.width, image.height)
        
        # Prepare input data
        tokens = []
        bboxes = []
        
        for box in boxes:
            for word in box.words:
                tokens.append(word)
                # Normalize bbox coordinates to 0-1000 range
                normalized_bbox = normalize_box(
                    [box.x, box.y, box.x + box.w, box.y + box.h], 
                    width, 
                    height
                )
                bboxes.append(normalized_bbox)
        
        # Prepare for model
        encoding = self.processor(
            image, 
            tokens, 
            boxes=bboxes, 
            return_tensors="pt",
            truncation=True,
            padding="max_length"
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Get logits and predictions
        logits = outputs.logits
        predictions = logits.argmax(-1).squeeze().tolist()
        
        # Convert to list if it's a single prediction
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        # Map token-level predictions back to word-level
        word_predictions = []
        word_idx = 0
        for i, pred in enumerate(predictions):
            if i < len(tokens) and pred != -100:
                word_predictions.append(self.id2label[pred])
                word_idx += 1
        
        # Map word-level predictions back to box-level
        box_predictions = []
        word_idx = 0
        
        for box in boxes:
            # Get predictions for all words in this box
            box_pred_labels = []
            for _ in box.words:
                if word_idx < len(word_predictions):
                    box_pred_labels.append(word_predictions[word_idx])
                    word_idx += 1
            
            # Use majority voting to determine the box label
            if box_pred_labels:
                # Remove B- or I- prefix if present
                processed_labels = [label[2:] if label.startswith(("B-", "I-")) else label for label in box_pred_labels]
                
                # Find the most common label (excluding "O")
                label_counts = {}
                for label in processed_labels:
                    if label != "O":
                        label_counts[label] = label_counts.get(label, 0) + 1
                
                if label_counts:
                    # Get the label with the highest count
                    box_label = max(label_counts.items(), key=lambda x: x[1])[0]
                else:
                    box_label = "O"
                
                box_predictions.append(box_label)
            else:
                box_predictions.append("O")
        
        return box_predictions

    def load_model(self, model_dir):
        """
        Load a trained model from a directory
        
        Args:
            model_dir: Directory containing the model
            
        Returns:
            Loaded model
        """
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        
        # Get label mappings from the model config
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        return self.model