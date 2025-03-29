# LayoutLMv3 PDF Form Annotator

A comprehensive application for annotating PDF forms and extracting structured data using Microsoft's LayoutLMv3 model.

## Features

- **PDF Form Annotation**: Easily mark and label form fields in PDF documents
- **LayoutLMv3 Integration**: Leverage state-of-the-art document understanding AI
- **Training Interface**: Fine-tune the model on your specific form types
- **Batch Processing**: Process multiple forms in batch mode
- **Form Field Extraction**: Automatically extract values from filled PDF forms

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.10.0 or higher
- Tesseract OCR (for text extraction)

### Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/layoutlm-form-annotator.git
   cd layoutlm-form-annotator
   ```

2. Install dependencies:

   ```
   pip install -e .
   ```

   Or directly:

   ```
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - Windows: Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt install tesseract-ocr`
   - macOS: `brew install tesseract`

## Usage

### Running the Application

```
python form_annotator.py
```

Or if installed as a package:

```
form-annotator
```

### Workflow

1. **Load a PDF**: Open a PDF form using the File menu
2. **Annotate Fields**: Mark form fields by drawing rectangles and labeling them
3. **Save Annotations**: Save your annotations to reuse later
4. **Train the Model**: Collect training data and fine-tune the model for your forms
5. **Extract Data**: Process new filled-in forms to extract field values

## Training Your Own Model

For best results, train the model on your specific form types:

1. Annotate fields on multiple examples of your forms
2. Collect training data by assigning values to each field
3. Train the model using the Training tab
4. Save your custom model for future use

## Project Structure

```
layoutlm-form-annotator/
├── form_annotator.py       # Main application file
├── setup.py                # Package installation config
├── requirements.txt        # Dependencies
├── README.md               # This file
└── layoutlm_form_model/    # Default model directory (created after training)
```

## Advanced Features

- **OCR Preprocessing**: Optimize text extraction for difficult forms
- **Model Optimization**: Adjust training parameters for specific form types
- **PDF Preprocessing**: Handle complex form layouts
- **Batch Export**: Process multiple forms and export results

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft Research for the LayoutLMv3 model
- HuggingFace for the Transformers library
- PyMuPDF for PDF processing capabilities
