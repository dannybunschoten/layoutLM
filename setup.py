from setuptools import setup, find_packages

setup(
    name="layoutlm-form-annotator",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A PDF form annotation and extraction tool using LayoutLMv3",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/layoutlm-form-annotator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.15.0",
        "pillow>=8.0.0",
        "pymupdf>=1.18.0",
        "numpy>=1.19.0",
        "pytesseract>=0.3.8",
        "opencv-python>=4.5.0",
        "tensorboard>=2.8.0",
        "tqdm>=4.62.0",
    ],
    entry_points={
        "console_scripts": [
            "form-annotator=form_annotator.app:main",
        ],
    },
)
