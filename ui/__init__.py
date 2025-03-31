"""
UI package for PDF Information Extractor
"""

from pdf_info_extractor.ui.app import PDFInfoExtractorApp
from pdf_info_extractor.ui.annotation_tab import AnnotationTab
from pdf_info_extractor.ui.training_tab import TrainingTab
from pdf_info_extractor.ui.inference_tab import InferenceTab

__all__ = ["PDFInfoExtractorApp", "AnnotationTab", "TrainingTab", "InferenceTab"]
