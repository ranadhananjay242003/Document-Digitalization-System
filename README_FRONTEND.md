# OCR Web Application Frontend

A modern, responsive web interface for the advanced OCR pipeline that extracts text from handwritten documents using AI models including CRAFT and TrOCR.

## Features

- **Drag & Drop Interface**: Easy image upload with drag and drop support
- **Multiple OCR Pipelines**: Choose from Advanced, Optimized, or Basic processing modes
- **Real-time Progress**: Live progress updates during text extraction
- **Image Preview**: Preview selected images before processing
- **PDF Export**: Download extracted text as formatted PDF documents
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Toast Notifications**: User-friendly feedback messages

## Setup Instructions

### 1. Install Dependencies

```powershell
# Create and activate virtual environment
python -m venv ocr-env
ocr-env\Scripts\Activate.ps1

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Download CRAFT Model

```powershell
# Create weights directory
mkdir craft_pytorch\weights

# Download the CRAFT model (craft_mlt_25k.pth) from:
# https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ
# Place it in craft_pytorch/weights/
```

### 3. Run the Web Application

```powershell
python app.py
```

The application will be available at: http://localhost:5000

## OCR Pipeline Options

### Advanced Pipeline
- **Best Quality**: Uses ensemble methods with multiple models
- **Features**: TrOCR + EasyOCR, deduplication, advanced post-processing
- **Use Case**: Highest accuracy for complex handwritten documents
- **Processing Time**: Slower (2-5 minutes)

### Optimized Pipeline
- **Balanced**: Good balance of speed and accuracy
- **Features**: TrOCR with optimized parameters
- **Use Case**: General purpose handwritten text extraction
- **Processing Time**: Medium (1-3 minutes)

### Basic Pipeline
- **Fastest**: Lightweight processing
- **Features**: TrOCR with basic preprocessing
- **Use Case**: Clear handwriting, quick processing needed
- **Processing Time**: Fast (30 seconds - 2 minutes)

## API Endpoints

The web application provides the following REST API endpoints:

### Upload Image
```
POST /upload
Content-Type: multipart/form-data

Parameters:
- file: Image file (PNG, JPG, JPEG)
- pipeline: Pipeline type ('advanced', 'optimized', 'basic')

Response:
{
  "task_id": "uuid-string",
  "message": "File uploaded successfully",
  "filename": "original-filename.jpg"
}
```

### Check Status
```
GET /status/{task_id}

Response:
{
  "status": "processing|completed|error",
  "progress": 0-100,
  "message": "Status message"
}
```

### Get Results
```
GET /result/{task_id}

Response:
{
  "lines": ["extracted", "text", "lines"],
  "line_count": 10,
  "pdf_path": "/path/to/generated.pdf",
  "processed_at": "2025-01-07T14:15:04"
}
```

### Download PDF
```
GET /download/{task_id}

Response: PDF file download
```

## File Structure

```
├── app.py                 # Flask web application
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── styles.css        # CSS styling
│   └── app.js           # JavaScript functionality
├── uploads/              # Temporary uploaded files
├── results/              # Generated PDF files
├── src/                  # OCR pipeline implementations
├── craft_pytorch/        # CRAFT model and utilities
└── requirements.txt      # Python dependencies
```

## Configuration

### Security
Update the secret key in `app.py`:
```python
app.secret_key = 'your-secure-random-secret-key'
```

### File Limits
- Maximum file size: 16MB (configurable in `app.py`)
- Supported formats: PNG, JPG, JPEG
- Processing timeout: 10 minutes

### Storage
- Uploaded files are stored temporarily in `uploads/`
- Generated PDFs are stored in `results/`
- Files are automatically cleaned up after processing

## Troubleshooting

### Common Issues

1. **CRAFT Model Not Found**
   - Download `craft_mlt_25k.pth` and place in `craft_pytorch/weights/`
   - Ensure file is exactly 67MB

2. **Out of Memory Errors**
   - Large images are automatically scaled to 1600px max dimension
   - Use CPU processing if GPU memory is insufficient

3. **Slow Processing**
   - Use "Optimized" or "Basic" pipeline for faster results
   - Ensure adequate RAM (8GB+ recommended)

4. **No Text Detected**
   - Ensure image has clear, readable handwriting
   - Try different pipeline options
   - Check image quality and contrast

### Performance Tips

- Use GPU acceleration when available (CUDA)
- Process images in good lighting with high contrast
- Avoid very large images (>5MB) for faster processing
- Close other applications to free up system resources

## Browser Compatibility

- **Recommended**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Features**: HTML5 File API, Fetch API, CSS Grid, Flexbox
- **Mobile**: iOS Safari 14+, Chrome Mobile 90+

## Development

To customize the interface:

1. **Styling**: Edit `static/styles.css`
2. **Functionality**: Edit `static/app.js`
3. **Layout**: Edit `templates/index.html`
4. **Backend**: Edit `app.py`

### Adding New Features

The modular structure makes it easy to add new features:
- OCR pipelines: Add to `src/` directory
- UI components: Add to templates and static files
- API endpoints: Add to `app.py`

## License

This project uses the same license as the original CRAFT implementation. See the main project documentation for details.
