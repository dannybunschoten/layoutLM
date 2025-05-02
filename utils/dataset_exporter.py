import os
import json
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Image as ImageFeature

class DatasetExporter:
    """Utility class for exporting labeled data to Hugging Face dataset format"""
    
    @staticmethod
    def export_dataset(documents, output_dir):
        """
        Export labeled documents to a Hugging Face dataset format
        
        Args:
            documents: List of Document objects with labeled text boxes
            
        Returns:
            Path to the dataset directory
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Collect all examples
        example_dicts = []
        
        # Get all unique labels to build the label list
        all_labels = set(["O"])
        
        for doc_idx, doc in enumerate(documents):
            for page_idx, boxes in enumerate(doc.page_boxes):
                if not boxes:  # Skip pages with no boxes
                    continue
                
                # Create a unique ID for this example
                example_id = f"{doc.filename.replace('.pdf', '')}_{page_idx}"
                
                # Save the image - convert to standard format to avoid PIL issues
                image = doc.images[page_idx]
                image_path = os.path.join(images_dir, f"{example_id}.png")
                
                # Convert to RGB if needed and save as PNG
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(image_path, "PNG")
                
                # Collect all tokens, bboxes, and labels
                tokens = []
                bboxes = []
                ner_tags = []
                
                for box in boxes:
                    if box.label != "O":
                        # Track unique labels
                        all_labels.add(f"B-{box.label}")
                        all_labels.add(f"I-{box.label}")
                    
                    for word_idx, word in enumerate(box.words):
                        tokens.append(word)
                        # Convert [x, y, w, h] to [x1, y1, x2, y2]
                        bbox = [box.x, box.y, box.x + box.w, box.y + box.h]
                        bboxes.append(bbox)
                        
                        # Check if this is first word (B-) or continuation (I-)
                        if box.label != "O":
                            if word_idx == 0:
                                ner_tags.append(f"B-{box.label}")
                            else:
                                ner_tags.append(f"I-{box.label}")
                        else:
                            ner_tags.append("O")
                
                # Store image path instead of PIL image
                example = {
                    "id": example_id,
                    "tokens": tokens,
                    "bboxes": bboxes,
                    "ner_tags": ner_tags,
                    "image_path": image_path  # Store path, not PIL object
                }
                
                example_dicts.append(example)
        
        # Sort labels, keeping "O" first
        sorted_labels = ["O"] + sorted([l for l in all_labels if l != "O"])
        
        # Create the dataset files manually
        DatasetExporter._create_dataset_files(example_dicts, sorted_labels, output_dir)
        
        # Create loading script
        DatasetExporter.create_dataset_loading_script(output_dir)
        
        return output_dir
    
    @staticmethod
    def _create_dataset_files(examples, label_names, output_dir):
        """Create dataset files manually instead of using Dataset.from_pandas"""
        # Create a data directory structure for the dataset
        data_dir = os.path.join(output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create the data files
        tokens_file = os.path.join(data_dir, "tokens.txt")
        bboxes_file = os.path.join(data_dir, "bboxes.txt")
        ner_tags_file = os.path.join(data_dir, "ner_tags.txt")
        image_paths_file = os.path.join(data_dir, "image_paths.txt")
        ids_file = os.path.join(data_dir, "ids.txt")
        
        # Write data to files
        with open(tokens_file, 'w') as f_tokens, \
             open(bboxes_file, 'w') as f_bboxes, \
             open(ner_tags_file, 'w') as f_ner_tags, \
             open(image_paths_file, 'w') as f_image_paths, \
             open(ids_file, 'w') as f_ids:
            
            for example in examples:
                # Write tokens (joined by space)
                f_tokens.write(' '.join(example['tokens']) + '\n')
                
                # Write bboxes (as JSON string)
                f_bboxes.write(json.dumps(example['bboxes']) + '\n')
                
                # Write ner_tags (joined by space)
                f_ner_tags.write(' '.join(example['ner_tags']) + '\n')
                
                # Write image paths
                f_image_paths.write(example['image_path'] + '\n')
                
                # Write ids
                f_ids.write(example['id'] + '\n')
        
        # Create dataset script file
        with open(os.path.join(output_dir, "dataset_script.py"), "w") as f:
            f.write(DatasetExporter._create_dataset_script(label_names))
        
        # Create dataset_info.json
        with open(os.path.join(output_dir, "dataset_infos.json"), "w") as f:
            dataset_info = {
                "document_layout": {
                    "description": "Custom document dataset exported from PDF OCR Tool",
                    "citation": "",
                    "homepage": "",
                    "license": "",
                    "features": {
                        "id": {"dtype": "string", "_type": "Value"},
                        "tokens": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"},
                        "bboxes": {"feature": {"feature": {"dtype": "int64", "_type": "Value"}, "_type": "Sequence", "length": 4}, "_type": "Sequence"},
                        "ner_tags": {"feature": {"num_classes": len(label_names), "names": label_names, "_type": "ClassLabel"}, "_type": "Sequence"},
                        "image": {"_type": "Image"}
                    },
                    "splits": {
                        "train": {"name": "train", "num_bytes": 0, "num_examples": len(examples), "dataset_name": "document_layout"}
                    },
                    "download_checksums": {},
                    "download_size": 0,
                    "dataset_size": 0,
                    "version": {"version_str": "1.0.0", "description": "", "major": 1, "minor": 0, "patch": 0}
                }
            }
            json.dump(dataset_info, f, indent=2)
    
    @staticmethod
    def _create_dataset_script(label_names):
        """Create a dataset script file"""
        return f"""
import os
import json
import datasets

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = \"""
Custom document dataset exported from PDF OCR Tool
\"""

_CITATION = \"""
@inproceedings{{
  author = {{Custom Dataset}},
  title = {{Document Layout Analysis Dataset}},
  year = {{2023}}
}}
\"""

class DocumentLayoutConfig(datasets.BuilderConfig):
    \"""Builder config for the Document Layout dataset\"""

    def __init__(self, **kwargs):
        \"""BuilderConfig for DocumentLayout.
        Args:
          **kwargs: keyword arguments forwarded to super.
        \"""
        super(DocumentLayoutConfig, self).__init__(**kwargs)


class DocumentLayout(datasets.GeneratorBasedBuilder):
    \"""Document Layout dataset.\"""

    BUILDER_CONFIGS = [
        DocumentLayoutConfig(name="document_layout", version=datasets.Version("1.0.0"), description="Document Layout dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {{
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"), length=4)),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names={label_names}
                        )
                    ),
                    "image": datasets.Image(),
                }}
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={{"data_dir": os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")}},
            ),
        ]

    def _generate_examples(self, data_dir):
        \"""Yields examples from the dataset.\"""
        logger.info("Generating examples from = %s", data_dir)
        
        # Load files
        with open(os.path.join(data_dir, "tokens.txt"), "r") as f_tokens, \\
             open(os.path.join(data_dir, "bboxes.txt"), "r") as f_bboxes, \\
             open(os.path.join(data_dir, "ner_tags.txt"), "r") as f_ner_tags, \\
             open(os.path.join(data_dir, "image_paths.txt"), "r") as f_image_paths, \\
             open(os.path.join(data_dir, "ids.txt"), "r") as f_ids:
            
            tokens_lines = f_tokens.readlines()
            bboxes_lines = f_bboxes.readlines()
            ner_tags_lines = f_ner_tags.readlines()
            image_paths_lines = f_image_paths.readlines()
            ids_lines = f_ids.readlines()
            
            for i, (tokens, bboxes, ner_tags, image_path, id_) in enumerate(
                zip(tokens_lines, bboxes_lines, ner_tags_lines, image_paths_lines, ids_lines)
            ):
                tokens = tokens.strip().split()
                bboxes = json.loads(bboxes.strip())
                ner_tags = ner_tags.strip().split()
                image_path = image_path.strip()
                id_ = id_.strip()
                
                yield i, {{
                    "id": id_,
                    "tokens": tokens,
                    "bboxes": bboxes,
                    "ner_tags": ner_tags,
                    "image": image_path,
                }}
"""
    
    @staticmethod
    def create_dataset_loading_script(output_dir):
        """
        Create a Python script to load the dataset
        
        Args:
            output_dir: Directory where the dataset is saved
            
        Returns:
            Path to the loading script
        """
        script_content = f"""from datasets import load_dataset

# Load the local dataset
dataset = load_dataset("{output_dir}")

# If you've uploaded to Hugging Face Hub, you can use:
# dataset = load_dataset("username/dataset_name")

# To show dataset information
print(dataset)
print(dataset["train"].features)

# Example usage
example = dataset["train"][0]
print(example.keys())
print(f"Number of tokens: {{len(example['tokens'])}}")
print(f"First few tokens: {{example['tokens'][:5]}}")
print(f"First few bounding boxes: {{example['bboxes'][:5]}}")
print(f"First few NER tags: {{example['ner_tags'][:5]}}")
"""
        
        script_path = os.path.join(output_dir, "load_dataset.py")
        with open(script_path, "w") as f:
            f.write(script_content)
        
        return script_path