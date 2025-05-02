import os
import numpy as np
import torch
import json
import copy
from transformers import (
    AutoProcessor,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    TrainerCallback
)
from datasets import Dataset, Features, Sequence, Value, Features, Sequence, Value, ClassLabel, Image as DsImage
from seqeval.metrics import f1_score, precision_score, recall_score

class MetricsCallback(TrainerCallback):
    """Callback to collect metrics during training"""
    def __init__(self):
        self.metrics_history = []
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Make a deep copy to avoid reference issues
        if metrics:
            self.metrics_history.append(copy.deepcopy(metrics))
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print("Epoch completed.")
        # If we're not evaluating every epoch, we still want to track train loss
        if not self.metrics_history or len(self.metrics_history) < state.epoch:
            # Check if log_history exists and has entries
            if hasattr(state, 'log_history') and state.log_history:
                metrics = {"train_loss": state.log_history[-1].get("loss", 0)}
                self.metrics_history.append(metrics)
            else:
                # Add a placeholder if there's no log_history
                metrics = {"train_loss": 0.0}
                self.metrics_history.append(metrics)

class ModelTrainer:
    """Simplified trainer for document layout models based on the working notebook approach"""
    
    def __init__(self, output_dir="trained_model"):
        self.output_dir = output_dir
        self.processor = None
        self.model = None
        self.label_list = None
        self.id2label = None
        self.label2id = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def prepare_dataset_from_documents(self, documents):
        """Convert Document objects to a dataset format for training"""
        train_examples = []
        
        for doc in documents:
            for page_idx, boxes in enumerate(doc.page_boxes):
                if not boxes:  # Skip pages with no boxes
                    continue
                
                # Get the page image
                image = doc.images[page_idx]
                img_width, img_height = image.size
                
                # Extract data for this page
                tokens = []
                bboxes = []
                ner_tags = []
                
                for box in boxes:
                    # Only include boxes with text
                    if box.text.strip():
                        tokens.append(box.text)
                        
                        # Convert box coordinates to LayoutLM format (normalized 0-1000)
                        x, y, w, h = box.x, box.y, box.w, box.h
                        x1, y1 = x, y
                        x2, y2 = x + w, y + h
                        
                        # Normalize coordinates to 0-1000 range
                        bbox = [
                            int(x1 / img_width * 1000),
                            int(y1 / img_height * 1000),
                            int(x2 / img_width * 1000),
                            int(y2 / img_height * 1000)
                        ]
                        
                        bboxes.append(bbox)
                        ner_tags.append(box.label)
                
                # Only add examples with content
                if tokens:
                    train_examples.append({
                        "id": f"{doc.filename}_page{page_idx}",
                        "tokens": tokens,
                        "bboxes": bboxes,
                        "ner_tags": ner_tags,
                        "image": image
                    })
        
        if not train_examples:
            raise ValueError("No valid examples found in the provided documents")
            
        return train_examples
    
    def train(self, documents, val_split=0.2, 
              model_name="microsoft/layoutlmv3-base", 
              batch_size=2, learning_rate=5e-5, epochs=3):
        """Train the model on document data"""
        # Prepare examples from documents
        examples = self.prepare_dataset_from_documents(documents)
        print(f"Created {len(examples)} examples from documents")
        
        # Collect all unique labels
        label_set = set()
        for example in examples:
            label_set.update(example["ner_tags"])
        
        # Create label list and mappings
        self.label_list = ['O'] + sorted(label_set - {'O'})
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        print(f"Found labels: {self.label_list}")
        
        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
        
        # Convert examples to ClassLabel format for ner_tags
        for example in examples:
            example["ner_tags"] = [self.label2id[tag] for tag in example["ner_tags"]]
        
        # Create dataset
        features = Features({
            "id": Value("string"),
            "tokens": Sequence(Value("string")),
            "bboxes": Sequence(Sequence(Value("int64"))),
            "ner_tags": Sequence(ClassLabel(names=self.label_list)),
            "image": DsImage(decode=True)
        })
        
        dataset = Dataset.from_list(examples, features=features)
        
        # Split into train and validation sets
        if val_split > 0:
            split = dataset.train_test_split(test_size=val_split)
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            train_dataset = dataset
            eval_dataset = None
        
        print(f"Train dataset: {len(train_dataset)} examples")
        if eval_dataset:
            print(f"Eval dataset: {len(eval_dataset)} examples")
        
        # Define preprocessing function
        def preprocess_data(examples):
            images = examples["image"]
            words = examples["tokens"]
            boxes = examples["bboxes"]
            word_labels = examples["ner_tags"]
            
            encoding = self.processor(
                images=images,
                text=words,
                boxes=boxes,
                word_labels=word_labels,
                truncation=True,
                padding="max_length"
            )
            
            return encoding
        
        # Process datasets
        train_dataset = train_dataset.map(
            preprocess_data,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                preprocess_data,
                batched=True,
                remove_columns=eval_dataset.column_names,
            )
        
        # Set PyTorch format
        train_dataset.set_format("torch")
        if eval_dataset:
            eval_dataset.set_format("torch")
        
        # Define compute_metrics function
        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)
            
            # Remove ignored index (special tokens)
            true_predictions = [
                [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            
            results = {
                "precision": precision_score(true_labels, true_predictions),
                "recall": recall_score(true_labels, true_predictions),
                "f1": f1_score(true_labels, true_predictions),
            }
            
            return results
        
        # Initialize the model
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            max_steps=1000,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1" if eval_dataset else None,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            report_to="none",
        )
        
        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics if eval_dataset else None,
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        
        # Save label mappings
        with open(os.path.join(self.output_dir, "label_mappings.json"), "w") as f:
            json.dump({
                "id2label": self.id2label,
                "label2id": self.label2id,
                "label_list": self.label_list
            }, f)
        
        # Return training results
        return {
            "model_path": self.output_dir,
            "training_loss": train_result.metrics.get("training_loss", 0),
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        }
    
    def predict(self, document, page_idx):
        """Run inference on a document page"""
        if not self.model or not self.processor:
            raise ValueError("Model not trained or loaded yet")
        
        # Get the page image
        image = document.images[page_idx]
        
        # Extract text boxes if available
        if document.page_boxes[page_idx]:
            boxes = document.page_boxes[page_idx]
            tokens = [box.text for box in boxes]
            
            # Convert box coordinates to LayoutLM format
            img_width, img_height = image.size
            bboxes = []
            for box in boxes:
                x, y, w, h = box.x, box.y, box.w, box.h
                bboxes.append([
                    int(x / img_width * 1000),
                    int(y / img_height * 1000),
                    int((x + w) / img_width * 1000),
                    int((y + h) / img_height * 1000)
                ])
            
            # Process using the processor
            encoding = self.processor(
                image,
                text=tokens,
                boxes=bboxes,
                return_tensors="pt",
                return_offsets_mapping=True
            )
            offset_mapping = encoding.pop("offset_mapping")
            word_ids = offset_mapping.word_ids()
        else:
            # Use processor's OCR
            encoding = self.processor(
                image, 
                return_tensors="pt",
                return_offsets_mapping=True
            )
            offset_mapping = encoding.pop("offset_mapping")
            word_ids = offset_mapping.word_ids()
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Process predictions
        logits = outputs.logits
        predictions = logits.argmax(-1).squeeze().tolist()
        
        # Map predictions back to words
        results = []
        word_to_prediction = {}
        
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                # Only consider first token of each word
                if idx == 0 or word_ids[idx-1] != word_id:
                    label_id = predictions[idx]
                    label = self.id2label.get(label_id, "O")
                    word_to_prediction[word_id] = label
        
        # Format results
        if document.page_boxes[page_idx]:
            # Use existing text boxes
            for i, box in enumerate(document.page_boxes[page_idx]):
                if i in word_to_prediction:
                    results.append({
                        "text": box.text,
                        "box": [box.x, box.y, box.w, box.h],
                        "label": word_to_prediction[i]
                    })
        else:
            # Use processor's OCR results
            unique_word_ids = set(word_id for word_id in word_ids if word_id is not None)
            for word_id in unique_word_ids:
                if word_id in word_to_prediction:
                    # Get token indices for this word
                    indices = [i for i, wid in enumerate(word_ids) if wid == word_id]
                    
                    # Get bbox from first token
                    box = encoding.bbox[0][indices[0]].tolist()
                    
                    # Convert normalized coordinates back to pixels
                    img_width, img_height = image.size
                    x1 = int(box[0] * img_width / 1000)
                    y1 = int(box[1] * img_height / 1000)
                    x2 = int(box[2] * img_width / 1000)
                    y2 = int(box[3] * img_height / 1000)
                    
                    # Get text 
                    token_ids = [encoding.input_ids[0][i].item() for i in indices]
                    text = self.processor.tokenizer.decode(token_ids)
                    
                    results.append({
                        "text": text,
                        "box": [x1, y1, x2-x1, y2-y1],  # [x, y, w, h] format
                        "label": word_to_prediction[word_id]
                    })
        
        return results
    
    def load_model(self, model_dir):
        """Load a trained model from a directory"""
        try:
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(model_dir)
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
            
            # Get label mappings
            label_path = os.path.join(model_dir, "label_mappings.json")
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    mappings = json.load(f)
                    self.id2label = {int(k): v for k, v in mappings["id2label"].items()}
                    self.label2id = mappings["label2id"]
                    self.label_list = mappings["label_list"]
            else:
                # Fallback to model config
                self.id2label = self.model.config.id2label
                self.label2id = self.model.config.label2id
                self.label_list = list(self.id2label.values())
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False