from flask import Flask, request, jsonify, render_template, send_file, flash, redirect, url_for
import os
import uuid
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import threading
import time

# Import our OCR pipelines
from src.advanced_ocr_pipeline import AdvancedOCRPipeline
from src.optimized_ocr_pipeline import OptimizedOCRPipeline
from src.run_ocr_pipeline import LineBasedOCRPipeline

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this to a random secret key
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Store processing status
processing_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_ocr_async(file_path, task_id, pipeline_type='advanced'):
    """Process OCR in background thread"""
    try:
        processing_status[task_id] = {
            'status': 'processing',
            'progress': 10,
            'message': 'Initializing OCR pipeline...'
        }
        
        # Initialize the selected pipeline
        if pipeline_type == 'advanced':
            pipeline = AdvancedOCRPipeline(verbose=False, enable_easyocr=False)
        elif pipeline_type == 'optimized':
            pipeline = OptimizedOCRPipeline()
        else:  # basic
            pipeline = LineBasedOCRPipeline(verbose=False)
        
        processing_status[task_id].update({
            'progress': 30,
            'message': 'Models loaded, processing image...'
        })
        
        # Extract text
        extracted_lines = pipeline.extract_text_from_image(file_path)
        
        processing_status[task_id].update({
            'progress': 80,
            'message': 'Generating PDF...'
        })
        
        # Save to PDF
        pdf_path = pipeline.save_to_pdf(extracted_lines, file_path, RESULTS_FOLDER)
        
        # Prepare results
        result_data = {
            'lines': extracted_lines,
            'line_count': len(extracted_lines),
            'pdf_path': pdf_path,
            'processed_at': datetime.now().isoformat()
        }
        
        processing_status[task_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Processing completed successfully!',
            'result': result_data
        }
        
    except Exception as e:
        processing_status[task_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error processing image: {str(e)}',
            'error': str(e)
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    pipeline_type = request.form.get('pipeline', 'advanced')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported. Please use PNG, JPG, or JPEG.'}), 400
    
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_{filename}")
        file.save(file_path)
        
        # Start background processing
        thread = threading.Thread(
            target=process_ocr_async,
            args=(file_path, task_id, pipeline_type)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'task_id': task_id,
            'message': 'File uploaded successfully. Processing started.',
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(processing_status[task_id])

@app.route('/result/<task_id>')
def get_result(task_id):
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    
    status = processing_status[task_id]
    
    if status['status'] != 'completed':
        return jsonify({'error': 'Task not completed yet'}), 400
    
    return jsonify(status['result'])

@app.route('/download/<task_id>')
def download_pdf(task_id):
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    
    status = processing_status[task_id]
    
    if status['status'] != 'completed':
        return jsonify({'error': 'Task not completed yet'}), 400
    
    pdf_path = status['result']['pdf_path']
    
    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({'error': 'PDF file not found'}), 404
    
    return send_file(pdf_path, as_attachment=True, download_name=f"ocr_result_{task_id}.pdf")

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_tasks': len([k for k, v in processing_status.items() if v['status'] == 'processing'])
    })

# Cleanup old tasks periodically
def cleanup_old_tasks():
    """Remove tasks older than 1 hour"""
    current_time = time.time()
    old_tasks = []
    
    for task_id, status in processing_status.items():
        # Simple cleanup - remove completed tasks older than 1 hour
        if status['status'] in ['completed', 'error']:
            old_tasks.append(task_id)
    
    # Keep only recent tasks (simple approach)
    if len(old_tasks) > 10:
        for task_id in old_tasks[:-5]:  # Keep last 5
            if task_id in processing_status:
                del processing_status[task_id]

if __name__ == '__main__':
    print("Starting OCR Web Application...")
    print("Available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
