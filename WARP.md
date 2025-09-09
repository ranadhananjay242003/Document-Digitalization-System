# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an advanced OCR (Optical Character Recognition) pipeline for handwritten text, specifically designed to extract text from images and convert them to PDF format. The project combines multiple deep learning models including CRAFT (text detection) and TrOCR (text recognition) with EasyOCR as a fallback option.

## Architecture

### Core Components

1. **CRAFT Text Detection** (`craft_pytorch/`): Handles text region detection in images
   - Uses pretrained model `craft_mlt_25k.pth` (not included in repo - must be downloaded)
   - Provides character-level and word-level text region detection

2. **OCR Pipelines** (`src/`): Multiple implementations with different approaches
   - `advanced_ocr_pipeline.py`: Full-featured pipeline with ensemble methods and deduplication
   - `optimized_ocr_pipeline.py`: Performance-optimized version with simpler processing
   - `run_ocr_pipeline.py`: Basic line-based OCR pipeline
   - `improved_ocr_pipeline.py`: Enhanced version with better preprocessing
   - `test.py`: Testing pipeline with extensive post-processing

3. **Text Recognition**: Uses Microsoft TrOCR models
   - Prefers `microsoft/trocr-base-handwritten` for speed
   - Falls back to `microsoft/trocr-large-handwritten` for accuracy
   - EasyOCR integration for ensemble approach

4. **Web Interface** (`app.py`, `templates/`, `static/`): Modern web application
   - Flask-based REST API with async processing
   - Responsive HTML/CSS/JavaScript frontend
   - Drag-and-drop file upload
   - Real-time progress tracking
   - PDF download functionality

## Development Commands

### Environment Setup
```powershell
# Create and activate virtual environment
python -m venv ocr-env
ocr-env\Scripts\Activate.ps1

# Install dependencies (CRAFT)
cd craft_pytorch
pip install -r requirements.txt

# Install additional dependencies for main pipelines
pip install torch torchvision transformers pillow opencv-python reportlab easyocr scikit-image scipy
```

### Model Setup
```powershell
# Create weights directory and download CRAFT model
mkdir craft_pytorch\weights
# Download craft_mlt_25k.pth from Google Drive link in craft_pytorch/README.md
```

### Running OCR Pipelines

#### Web Interface (Recommended)
```powershell
# Start the web application
python app.py
# Or use the batch script
start_web_app.bat

# Open browser to: http://localhost:5000
```

#### Command Line Interface
```powershell
# Run basic OCR pipeline
python src\run_ocr_pipeline.py

# Run advanced pipeline with verbose output
python -c "from src.advanced_ocr_pipeline import AdvancedOCRPipeline; pipeline = AdvancedOCRPipeline(verbose=True, enable_easyocr=False); pipeline.extract_text_from_image('hihi/sample.png')"

# Test CRAFT detection standalone
cd craft_pytorch
python test.py --trained_model=weights/craft_mlt_25k.pth --test_folder=../hihi/
```

### Testing
```powershell
# Test with sample images
python src\test.py

# Test optimized pipeline
python src\optimized_ocr_pipeline.py
```

## Key Implementation Details

### Line Detection Strategies
The project uses multiple text line detection approaches:
1. **Morphological operations**: Uses OpenCV kernels to detect horizontal text lines
2. **CRAFT-based grouping**: Groups CRAFT character detections into lines
3. **Fixed horizontal strips**: Fallback method that divides image into horizontal bands
4. **Projection-based detection**: Uses horizontal pixel projection to find text regions

### Text Processing Pipeline
1. **Preprocessing**: CLAHE enhancement, bilateral filtering, adaptive thresholding
2. **Line Segmentation**: Multiple strategies for robust line detection
3. **Text Recognition**: TrOCR with beam search and optimized generation parameters
4. **Post-processing**: OCR error correction, deduplication, text merging
5. **Output**: PDF generation with ReportLab

### Model Configuration
- **Device**: Auto-detects CUDA availability, falls back to CPU
- **Image Scaling**: Automatically scales large images to max 1600px dimension
- **TrOCR Parameters**: Optimized for handwritten text with repetition penalty and length constraints

## File Structure Context

- `hihi/`: Contains sample images for testing (sample.jpg, sample.png)
- `src/`: Main OCR pipeline implementations
- `craft_pytorch/`: CRAFT text detection module (PyTorch implementation)
- `templates/`: HTML templates for web interface
- `static/`: CSS and JavaScript files for web interface
- `uploads/`: Temporary storage for uploaded images
- `results/`: Generated PDF files from web interface
- `.vscode/`: VS Code configuration
- `app.py`: Flask web application
- `start_web_app.bat`: Windows startup script
- Generated outputs: `*_extracted_text.pdf`, `*_advanced_extracted_text.pdf`

## Common Issues & Solutions

### CRAFT Model Loading
- Ensure `craft_mlt_25k.pth` is downloaded and placed in `craft_pytorch/weights/`
- Model file should be ~67MB
- Handle "module." prefix stripping in state dict

### Memory Management
- Large images are automatically downscaled to prevent memory issues
- Use CPU fallback if CUDA memory is insufficient
- Consider batch processing for multiple images

### Text Quality
- Handwritten text works best with clear, high-contrast images
- Adjust CRAFT thresholds (`text_threshold`, `link_threshold`) for different image types
- Use ensemble approach combining TrOCR and EasyOCR for better accuracy

## Development Notes

- Project uses Windows-style path separators (`\`) - update for cross-platform compatibility if needed
- Virtual environment name `ocr-env` is excluded in .gitignore
- Large model files (*.pth, *.pkl, *.h5) are excluded from version control
- PDF generation includes metadata and proper formatting for document archival
