# PDF OCR Annotation and Training Tool

This application allows users to upload PDFs, run OCR on them using Tesseract, annotate text boxes, combine multiple boxes, train a LayoutLMv3 model on the annotations, and run inference on new PDFs.

## Features

- PDF upload and navigation
- OCR processing with Tesseract
- Text box annotation with custom labels
- Box combination for multi-part text entities
- LayoutLMv3 model training
- Inference on new PDFs

## Installation

### Prerequisites

- Python 3.7+ installed
- Tesseract OCR installed on your system

### Steps

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/pdf-ocr-annotation-tool.git
   cd pdf-ocr-annotation-tool
   ```

2. Create a virtual environment (recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR if not already installed:
   - On Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - On macOS: `brew install tesseract`
   - On Windows: Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

1. Start the application:

   ```
   python main.py
   ```

2. Using the application:
   - Click "Open PDF" to load a PDF document
   - Navigate through pages using the arrow buttons
   - Click "Run OCR" to detect text boxes
   - Select a box and assign a label using the dropdown
   - Use "Start Combine" to merge multiple boxes together
   - Save your annotations using "Save Annotations"
   - Train the model using "Train Model"
   - Run inference on new PDFs using "Run Inference"

## Workflow

1. **Annotation Phase**:

   - Load multiple PDFs
   - Run OCR on each page
   - Label the text boxes according to their content type
   - Combine boxes if needed
   - Save annotations

2. **Training Phase**:

   - Load your saved annotations
   - Train the LayoutLMv3 model on your annotated data

3. **Inference Phase**:
   - Load a new PDF
   - Run inference to automatically label text boxes

## Customization

- Add new labels as needed for your document types
- Adjust model training parameters in the code

## Troubleshooting

- If OCR results are poor, try adjusting the page rendering resolution
- For memory issues during training, reduce batch size in the training arguments

## License

This project is licensed under the MIT License - see the LICENSE file for details.
