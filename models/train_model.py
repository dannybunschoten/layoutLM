import os
from typing import Dict, List, Tuple, TypedDict, cast, Any
import numpy as np
import torch
import json
from transformers import (  # type: ignore
  AutoProcessor,
  LayoutLMv3ForTokenClassification,
  AutoModelForTokenClassification,
  TrainingArguments,
  Trainer,
  default_data_collator,
)
from datasets import Dataset, Features, Sequence, Value, ClassLabel, Image as DsImage  # type: ignore
from seqeval.metrics import f1_score, precision_score, recall_score  # type: ignore
from models.document import Document


class datasetClass(TypedDict):
  id: str
  tokens: List[str]
  bboxes: List[Tuple[int, int, int, int]]
  ner_tags: List[int]
  image: str


class ModelTrainer:
  def __init__(self, output_dir: str = "trained_model"):
    self.output_dir = output_dir
    self.processor: AutoProcessor | None = None
    self.model: AutoModelForTokenClassification | None = None
    self.id2label: Dict[int, str] = {}
    self.label2id: Dict[str, int] = {}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

  def prepare_dataset_from_documents(self, documents: List[Document]):
    """Convert Document objects to a dataset format for training"""
    train_examples: List[datasetClass] = []

    label2id: Dict[str, int] = {}

    for doc in documents:
      for page_idx, boxes in enumerate(doc.page_boxes):
        if not boxes:
          continue

        image = doc.images[page_idx]
        img_width, img_height = image.size

        tokens: List[str] = []
        bboxes: List[Tuple[int, int, int, int]] = []
        ner_tags: List[int] = []

        for box in boxes:
          if box.text.strip():
            tokens.append(box.text)

            x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2

            bbox = (
              int(x1 / img_width * 1000),
              int(y1 / img_height * 1000),
              int(x2 / img_width * 1000),
              int(y2 / img_height * 1000),
            )

            bboxes.append(bbox)
            if box.label not in label2id:
              label2id[box.label] = len(label2id)
            ner_tags.append(label2id[box.label])

        # Only add examples with content
        if tokens:
          train_examples.append(
            {
              "id": f"{doc.filename}_page{page_idx}",
              "tokens": tokens,
              "bboxes": bboxes,
              "ner_tags": ner_tags,
              "image": doc.image_paths[page_idx],
            }
          )

    if not train_examples:
      raise ValueError("No valid examples found in the provided documents")

    return (train_examples, label2id)

  def train(
    self,
    documents: List[Document],
    val_split: float = 0.25,
    model_name: str = "microsoft/layoutlmv3-base",
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    epochs: int = 3,
  ):
    """Train the model on document data"""
    examples, label2id = self.prepare_dataset_from_documents(documents)
    id2label = {v: k for k, v in label2id.items()}

    processor: AutoProcessor = AutoProcessor.from_pretrained(  # type: ignore
      model_name, apply_ocr=False
    )

    features = Features(
      {
        "id": Value("string"),
        "tokens": Sequence(Value("string")),
        "bboxes": Sequence(Sequence(Value("int64"))),
        "ner_tags": Sequence(ClassLabel(names=list(label2id.keys()))),
        "image": DsImage(),
      }
    )

    dataset = Dataset.from_list(cast(List[Dict[str, Any]], examples), features=features)  # type: ignore

    # Split into train and validation sets
    if val_split > 0:
      split = dataset.train_test_split(test_size=val_split)
      train_dataset = split["train"]
      eval_dataset = split["test"]
    else:
      train_dataset = dataset
      eval_dataset = None
      print("No validation split provided, using all data for training")

    def preprocess_data(examples: Any) -> Dict[str, Any]:
      images = examples["image"]
      words = examples["tokens"]
      boxes = examples["bboxes"]
      word_labels = examples["ner_tags"]

      if processor is None:
        raise ValueError(
          "Processor is not initialized. Ensure the processor is loaded or set before calling this method."
        )

      encoding = processor(  # type: ignore
        images=images,
        text=words,
        boxes=boxes,
        word_labels=word_labels,
        truncation=True,
        padding="max_length",
      )

      return encoding  # type: ignore

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
    train_dataset.set_format("torch")  # type: ignore
    if eval_dataset:
      eval_dataset.set_format("torch")  # type: ignore

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
    model = LayoutLMv3ForTokenClassification.from_pretrained(  # type: ignore
      model_name,
      id2label=id2label,
      label2id=label2id,
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
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      tokenizer=processor,
      data_collator=default_data_collator,
      compute_metrics=compute_metrics if eval_dataset else None,
    )

    # Train the model
    train_result = trainer.train()

    # Save the model
    trainer.save_model(self.output_dir)
    trainer.evaluate()
    processor.save_pretrained(self.output_dir)

    # Save label mappings
    with open(os.path.join(self.output_dir, "label_mappings.json"), "w") as f:
      json.dump(
        {"id2label": id2label, "label2id": label2id},
        f,
      )

    # Return training results
    return {
      "model_path": self.output_dir,
      "training_loss": train_result.metrics.get("training_loss", 0),  # type: ignore
      "train_runtime": train_result.metrics.get("train_runtime", 0),  # type: ignore
      "train_samples_per_second": train_result.metrics.get(  # type: ignore
        "train_samples_per_second", 0
      ),
    }

  def predict(self, document: Document, page_idx: int) -> List[Dict[str, Any]]:
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
      bboxes: List[Tuple[int, int, int, int]] = []
      for box in boxes:
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
        bboxes.append(
          (
            int(x1 / img_width * 1000),
            int(y1 / img_height * 1000),
            int(x2 / img_width * 1000),
            int(y2 / img_height * 1000),
          )
        )

      # Process using the processor
      encoding = self.processor(
        image,
        text=tokens,
        boxes=bboxes,
        return_tensors="pt",
        return_offsets_mapping=True,
      )
      offset_mapping = encoding.pop("offset_mapping")
      word_ids = offset_mapping.word_ids()
    else:
      print("No text boxes found for this page, using OCR results")
      return []

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
        if idx == 0 or word_ids[idx - 1] != word_id:
          label_id = predictions[idx]
          label = self.id2label.get(label_id, "O")
          word_to_prediction[word_id] = label

    # Format results
    if document.page_boxes[page_idx]:
      # Use existing text boxes
      for i, box in enumerate(document.page_boxes[page_idx]):
        if i in word_to_prediction:
          results.append(
            {
              "text": box.text,
              "box": [box.x, box.y, box.w, box.h],
              "label": word_to_prediction[i],
            }
          )

    return results

  def load_model(self, model_dir: str):
    """Load a trained model from a directory"""
    try:
      # Load processor and model
      self.model = AutoModelForTokenClassification.from_pretrained(model_dir)  # type: ignore

      # Get label mappings
      label_path = os.path.join(model_dir, "label_mappings.json")
      if os.path.exists(label_path):
        with open(label_path, "r") as f:
          mappings = json.load(f)
          self.id2label = {int(k): v for k, v in mappings["id2label"].items()}
          self.label2id = mappings["label2id"]
      else:
        raise ValueError(f"Label mappings not found in {model_dir}")

      return True
    except Exception as e:
      print(f"Error loading model: {e}")
      return False
