<<<<<<< HEAD
from flask import Flask, request, jsonify, render_template, send_file, flash, redirect, url_for
import os
import uuid
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import threading
import time
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Import our OCR pipelines
from src.advanced_ocr_pipeline import AdvancedOCRPipeline
from src.optimized_ocr_pipeline import OptimizedOCRPipeline
from src.run_ocr_pipeline import LineBasedOCRPipeline
from src.fast_ocr_pipeline import FastOCRPipeline

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

def create_simple_fallback_pdf(lines, image_path, output_folder):
    """Create a simple PDF as fallback when main PDF generation fails"""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    from datetime import datetime
    
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"fallback_ocr_{base_name}_{timestamp}.pdf"
    output_path = os.path.join(output_folder, output_filename)
    
    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph(f"OCR Results - {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Heading1']))
        
        for line in lines:
            if line.strip():
                # Simple text processing to avoid ReportLab issues
                clean_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(clean_line, styles['Normal']))
        
        doc.build(story)
        return output_path
    except Exception as e:
        print(f"[ERROR] Even fallback PDF failed: {e}")
        # Return text file as last resort
        txt_path = output_path.replace('.pdf', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"OCR Results - {datetime.now()}\n" + "=" * 50 + "\n\n")
            for line in lines:
                f.write(line + "\n")
        return txt_path

def process_ocr_async(file_path, task_id, pipeline_type='advanced'):
    """Process OCR in background thread with optimized progress updates"""
    start_time = time.time()
    max_total_time = 300  # 5 minutes maximum total time
    
    try:
        processing_status[task_id] = {
            'status': 'processing',
            'progress': 5,
            'message': 'Initializing OCR pipeline...'
        }
        
        # Quick progress update for model loading
        processing_status[task_id].update({
            'progress': 15,
            'message': 'Loading AI models...'
        })
        
        # Initialize the selected pipeline with optimizations
        if pipeline_type == 'advanced':
            # Enable EasyOCR as fallback to ensure results
            pipeline = AdvancedOCRPipeline(verbose=False, enable_easyocr=True)
        elif pipeline_type == 'optimized':
            pipeline = OptimizedOCRPipeline()
        elif pipeline_type == 'fast':
            pipeline = FastOCRPipeline()  # Ultra-fast EasyOCR only
        else:  # default to fast for maximum speed
            pipeline_type = 'fast'  # Ensure we track that we're using fast
            pipeline = FastOCRPipeline()  # Use fast by default for best performance
        
        processing_status[task_id].update({
            'progress': 25,
            'message': 'Models loaded, analyzing image...'
        })
        
        # Start text detection progress
        processing_status[task_id].update({
            'progress': 35,
            'message': 'Detecting text regions...'
        })
        
        # Start text extraction with progress tracking
        processing_status[task_id].update({
            'progress': 45,
            'message': 'Extracting text from image...'
        })
        
        # Create a function to update progress during extraction
        def update_extraction_progress(current_step, total_steps=100):
            progress = min(95, 45 + int((current_step / total_steps) * 40))
            processing_status[task_id].update({
                'progress': progress,
                'message': 'Processing text extraction...'
            })
        
        # Extract text with simple, reliable progress tracking
        print(f"[DEBUG] Starting text extraction for task {task_id} using {pipeline_type} pipeline")
        
        # Simple progress updates without complex threading
        processing_status[task_id].update({
            'progress': 50,
            'message': 'Processing text extraction...'
        })
        
        try:
            # Perform actual text extraction with timeout
            timeout_seconds = 60 if pipeline_type == 'fast' else 120  # 1 min for fast, 2 min for others
            
            # Progress update during extraction
            processing_status[task_id].update({
                'progress': 65,
                'message': 'Analyzing text content...'
            })
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(pipeline.extract_text_from_image, file_path)
                
                # Simple polling with progress updates
                start_time = time.time()
                while not future.done():
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        future.cancel()
                        raise Exception(f"Text extraction timed out after {timeout_seconds} seconds")
                    
                    # Update progress based on elapsed time
                    progress_percent = min(80, 65 + int((elapsed / timeout_seconds) * 15))
                    processing_status[task_id].update({
                        'progress': progress_percent,
                        'message': 'Processing text extraction...'
                    })
                    time.sleep(0.5)  # Check every 0.5 seconds
                
                extracted_lines = future.result()
                print(f"[DEBUG] Text extraction completed successfully for task {task_id}")
            
            # Immediately advance progress after extraction
            processing_status[task_id].update({
                'progress': 85,
                'message': 'Text extraction completed!'
            })
            
            print(f"[DEBUG] Extracted {len(extracted_lines)} lines of text for task {task_id}")
            
        except Exception as extraction_error:
            print(f"[ERROR] Text extraction failed for task {task_id}: {extraction_error}")
            
            # Try to provide fallback results instead of failing completely
            try:
                if pipeline_type == 'advanced' and hasattr(pipeline, 'easyocr_reader') and pipeline.easyocr_reader is not None:
                    print(f"[DEBUG] Trying EasyOCR fallback for task {task_id}")
                    # Try basic EasyOCR extraction as fallback
                    import cv2
                    image = cv2.imread(file_path)
                    if image is not None:
                        easy_result = pipeline.easyocr_reader.readtext(image, detail=1)
                        if easy_result:
                            fallback_lines = []
                            for result in easy_result:
                                if len(result) >= 3 and result[2] > 0.1:  # Low confidence threshold
                                    fallback_lines.append(result[1].strip())
                            
                            if fallback_lines:
                                extracted_lines = fallback_lines
                                print(f"[DEBUG] EasyOCR fallback found {len(extracted_lines)} lines for task {task_id}")
                            else:
                                extracted_lines = [f"Text extraction partially failed: {str(extraction_error)[:100]}..."]
                        else:
                            extracted_lines = [f"Text extraction failed but processing completed: {str(extraction_error)[:100]}..."]
                    else:
                        extracted_lines = [f"Could not read image file: {str(extraction_error)[:100]}..."]
                else:
                    extracted_lines = [f"Processing encountered an error: {str(extraction_error)[:100]}..."]
            except Exception as fallback_error:
                print(f"[ERROR] Fallback also failed for task {task_id}: {fallback_error}")
                extracted_lines = [f"Text extraction failed: {str(extraction_error)[:100]}..."]
            
            # Continue with PDF generation even if extraction had issues
            print(f"[DEBUG] Continuing with {len(extracted_lines)} lines despite extraction error")
        
        # Generate PDF with progress
        processing_status[task_id].update({
            'progress': 90,
            'message': 'Generating PDF document...'
        })
        
        print(f"[DEBUG] Starting PDF generation for task {task_id} with {len(extracted_lines)} lines")
        
        # Save to PDF with timeout to prevent hanging
        pdf_generation_start = time.time()
        try:
            pdf_path = pipeline.save_to_pdf(extracted_lines, file_path, RESULTS_FOLDER)
            pdf_time = time.time() - pdf_generation_start
            print(f"[DEBUG] PDF generation completed in {pdf_time:.2f} seconds for task {task_id}")
        except Exception as pdf_error:
            print(f"[ERROR] PDF generation failed for task {task_id}: {pdf_error}")
            # Create a simple fallback PDF
            pdf_path = create_simple_fallback_pdf(extracted_lines, file_path, RESULTS_FOLDER)
            print(f"[DEBUG] Fallback PDF created for task {task_id}")
        
        # Force completion to ensure it never gets stuck
        processing_status[task_id].update({
            'progress': 95,
            'message': 'Finalizing document...'
        })
        
        time.sleep(0.1)  # Brief pause
        
        processing_status[task_id].update({
            'progress': 98,
            'message': 'Almost done...'
        })
        
        # Prepare results
        result_data = {
            'lines': extracted_lines,
            'line_count': len(extracted_lines),
            'pdf_path': pdf_path,
            'processed_at': datetime.now().isoformat()
        }
        
        # Force final completion
        processing_status[task_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Processing completed successfully!',
            'result': result_data
        }
        
        print(f"[DEBUG] Task {task_id} completed successfully with {len(extracted_lines)} lines")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[ERROR] Task {task_id} failed after {elapsed_time:.2f} seconds: {e}")
        
        processing_status[task_id] = {
            'status': 'error',
            'progress': 100,  # Set to 100 so UI knows it's done
            'message': f'Error processing image: {str(e)}',
            'error': str(e)
        }
    
    # Final safety check - ensure we never leave a task in limbo
    finally:
        if task_id in processing_status:
            current_status = processing_status[task_id]
            if current_status.get('status') == 'processing' and current_status.get('progress', 0) < 100:
                # Force completion if still processing
                elapsed_time = time.time() - start_time
                print(f"[WARNING] Force completing task {task_id} after {elapsed_time:.2f} seconds")
                
                processing_status[task_id].update({
                    'status': 'completed' if current_status.get('progress', 0) > 50 else 'error',
                    'progress': 100,
                    'message': 'Processing completed (forced)' if current_status.get('progress', 0) > 50 else 'Processing failed (timeout)',
                })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    pipeline_type = request.form.get('pipeline', 'fast')  # Default to fast pipeline for speed
    
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

@app.route('/debug/status')
def debug_status():
    """Debug endpoint to see all task statuses"""
    return jsonify({
        'total_tasks': len(processing_status),
        'tasks': processing_status,
        'timestamp': datetime.now().isoformat()
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
    
    # Use a more robust server configuration
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Reduce verbose logging
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down OCR Web Application...")
    except Exception as e:
        print(f"Server error: {e}")
||||||| (empty tree)
=======
from flask import Flask, request, jsonify, render_template, send_file, flash, redirect, url_for
import os
import uuid
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import threading
import time
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Import our OCR pipelines
from src.advanced_ocr_pipeline import AdvancedOCRPipeline
from src.optimized_ocr_pipeline import OptimizedOCRPipeline
from src.run_ocr_pipeline import LineBasedOCRPipeline
from src.fast_ocr_pipeline import FastOCRPipeline

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

# Cache pipelines to avoid reloading models
pipeline_cache = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_simple_fallback_pdf(lines, image_path, output_folder):
    """Create a simple PDF as fallback when main PDF generation fails"""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    from datetime import datetime
    
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"fallback_ocr_{base_name}_{timestamp}.pdf"
    output_path = os.path.join(output_folder, output_filename)
    
    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph(f"OCR Results - {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Heading1']))
        
        for line in lines:
            if line.strip():
                # Simple text processing to avoid ReportLab issues
                clean_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(clean_line, styles['Normal']))
        
        doc.build(story)
        return output_path
    except Exception as e:
        print(f"[ERROR] Even fallback PDF failed: {e}")
        # Return text file as last resort
        txt_path = output_path.replace('.pdf', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"OCR Results - {datetime.now()}\n" + "=" * 50 + "\n\n")
            for line in lines:
                f.write(line + "\n")
        return txt_path

def warp_lines_preserving_layout(lines):
    """Generically warp wording to feel sudden/chaotic/accelerated while
    preserving the original line layout and whitespace exactly.
    Strategy:
      - Split each line into tokens and separators; replace only tokens
      - Use broad lexical mappings (verbs, nouns, adjectives) and keep casing
      - Never insert/remove spaces or line breaks
    """
    if not lines:
        return lines

    import re

    # Generic tone maps (single-token only to avoid altering spacing)
    VERB_MAP = {
        'see':'flash','seen':'blazed','be':'be','become':'snap','bring':'hurl','brings':'hurls',
        'face':'skid','faced':'skidding','end':'crash','falls':'plunges','fall':'plunge',
        'lack':'lack','lacks':'lacks','survive':'persist','remind':'jolt','reminder':'jolt',
        'move':'surge','moving':'surging','rise':'spike','rises':'spikes'
    }
    NOUN_MAP = {
        'support':'scaffold','reminder':'jolt','crisis':'whiplash','depth':'torque',
        'speed':'whirl','change':'pivot','world':'world','culture':'culture',
        'translation':'translation','commentary':'commentary','consciousness':'consciousness'
    }
    ADJ_MAP = {
        'main':'primary','great':'surging','oldest':'long‑running','present':'immediate',
        'highly':'hyper','activistic':'activistic','one':'one','sided':'sided',
        'sudden':'sudden','fast':'rapid','slow':'stalled','moral':'moral','political':'political'
    }
    ADV_MAP = {'suddenly':'abruptly','quickly':'rapidly','slowly':'haltingly'}

    def preserve_case(src, repl):
        if src.isupper():
            return repl.upper()
        if src[:1].isupper() and src[1:].islower():
            return repl.capitalize()
        return repl

    def warp_token(tok):
        # keep punctuation-only tokens
        if not re.search(r"[A-Za-z]", tok):
            return tok
        # strip leading/trailing punctuation but keep them
        m = re.match(r"(^[^A-Za-z]*)(.*?)([^A-Za-z]*$)", tok)
        lead, core, trail = m.group(1), m.group(2), m.group(3)
        base = core
        key = base.lower()
        rep = None
        if key in VERB_MAP: rep = VERB_MAP[key]
        elif key in NOUN_MAP: rep = NOUN_MAP[key]
        elif key in ADJ_MAP: rep = ADJ_MAP[key]
        elif key in ADV_MAP: rep = ADV_MAP[key]
        elif '-' in key:
            parts = key.split('-')
            mapped = [VERB_MAP.get(p) or NOUN_MAP.get(p) or ADJ_MAP.get(p) or ADV_MAP.get(p) or p for p in parts]
            rep = '-'.join(mapped)
        else:
            rep = base
        rep = preserve_case(base, rep)
        return f"{lead}{rep}{trail}"

    def warp_line(line):
        # Replace underscores with spaces but keep count
        s = line.replace('_',' ')
        parts = re.split(r"(\s+)", s)
        for i in range(0, len(parts), 2):  # tokens at even indices
            parts[i] = warp_token(parts[i])
        return ''.join(parts)

    return [warp_line(l) if isinstance(l,str) else l for l in lines]


def process_ocr_async(file_path, task_id, pipeline_type='advanced', warp=False, spellcheck=True):
    """Process OCR in background thread with optimized progress updates"""
    start_time = time.time()
    max_total_time = 300  # 5 minutes maximum total time
    
    try:
        processing_status[task_id] = {
            'status': 'processing',
            'progress': 5,
            'message': 'Initializing OCR pipeline...'
        }
        
        # Use cached pipeline if available (HUGE speed boost!)
        if pipeline_type in pipeline_cache:
            pipeline = pipeline_cache[pipeline_type]
            processing_status[task_id].update({
                'progress': 25,
                'message': 'Using cached models (fast!)...'
            })
        else:
            # Load models for first time
            processing_status[task_id].update({
                'progress': 15,
                'message': 'Loading AI models (first time)...'
            })
            
            # Initialize the selected pipeline with optimizations
            if pipeline_type == 'advanced':
                pipeline = AdvancedOCRPipeline(verbose=False, enable_easyocr=True)
            elif pipeline_type == 'optimized':
                pipeline = OptimizedOCRPipeline()
            elif pipeline_type == 'fast':
                pipeline = FastOCRPipeline()  # Ultra-fast EasyOCR only
            else:  # default to fast for maximum speed
                pipeline_type = 'fast'
                pipeline = FastOCRPipeline()
            
            # Cache for future requests
            pipeline_cache[pipeline_type] = pipeline
        
        processing_status[task_id].update({
            'progress': 25,
            'message': 'Models loaded, analyzing image...'
        })
        
        # Start text detection progress
        processing_status[task_id].update({
            'progress': 35,
            'message': 'Detecting text regions...'
        })
        
        # Start text extraction with progress tracking
        processing_status[task_id].update({
            'progress': 45,
            'message': 'Extracting text from image...'
        })
        
        # Create a progress callback function
        def progress_callback(progress, message):
            """Update progress during text extraction"""
            processing_status[task_id].update({
                'progress': min(95, int(45 + (progress / 100) * 40)),  # Map 0-100 to 45-85
                'message': message
            })
        
        # Extract text with simple, reliable progress tracking
        print(f"[DEBUG] Starting text extraction for task {task_id} using {pipeline_type} pipeline")
        
        try:
            # Perform actual text extraction with timeout
            timeout_seconds = 60 if pipeline_type == 'fast' else 120
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Pass progress callback if the pipeline supports it
                if pipeline_type == 'fast':
                    future = executor.submit(pipeline.extract_text_from_image, file_path, progress_callback)
                else:
                    future = executor.submit(pipeline.extract_text_from_image, file_path)
                
                # Simple polling with timeout
                extraction_start = time.time()
                
                while not future.done():
                    elapsed = time.time() - extraction_start
                    if elapsed > timeout_seconds:
                        print(f"[WARNING] Extraction timeout after {timeout_seconds}s, cancelling...")
                        future.cancel()
                        raise Exception(f"Text extraction timed out after {timeout_seconds} seconds")
                    
                    # Only update if pipeline doesn't have its own progress callback
                    if pipeline_type != 'fast':
                        # Smooth progress from 45 to 85
                        progress_percent = min(85, int(45 + (elapsed / timeout_seconds) * 40))
                        processing_status[task_id].update({
                            'progress': progress_percent,
                            'message': 'Processing text extraction...'
                        })
                    
                    time.sleep(0.5)  # Check every 0.5 seconds
                
                extracted_lines = future.result(timeout=5)  # 5 second timeout for getting result
                print(f"[DEBUG] Text extraction completed successfully for task {task_id}")
            
            # Optionally disable spellchecking inside pipelines
            try:
                if not spellcheck and hasattr(pipeline, 'spell_checker'):
                    pipeline.spell_checker = None
                    print(f"[DEBUG] Spellcheck disabled for task {task_id}")
            except Exception:
                pass
            # Set simple corrections toggle if available
            try:
                if hasattr(pipeline, 'enable_simple_corrections'):
                    pipeline.enable_simple_corrections = simple_flag
            except Exception:
                pass

            # Optionally warp wording while preserving layout
            if warp and extracted_lines:
                try:
                    extracted_lines = warp_lines_preserving_layout(extracted_lines)
                    print(f"[DEBUG] Applied warp transformation for task {task_id}")
                except Exception as _e:
                    print(f"[WARNING] Warp transformation skipped: {_e}")
            
            # Immediately advance progress after extraction
            processing_status[task_id].update({
                'progress': 88,
                'message': 'Text extraction completed!'
            })
            
            print(f"[DEBUG] Extracted {len(extracted_lines)} lines of text for task {task_id}")
            
        except Exception as extraction_error:
            print(f"[ERROR] Text extraction failed for task {task_id}: {extraction_error}")
            
            # Try to provide fallback results instead of failing completely
            try:
                if pipeline_type == 'advanced' and hasattr(pipeline, 'easyocr_reader') and pipeline.easyocr_reader is not None:
                    print(f"[DEBUG] Trying EasyOCR fallback for task {task_id}")
                    # Try basic EasyOCR extraction as fallback
                    import cv2
                    image = cv2.imread(file_path)
                    if image is not None:
                        easy_result = pipeline.easyocr_reader.readtext(image, detail=1)
                        if easy_result:
                            fallback_lines = []
                            for result in easy_result:
                                if len(result) >= 3 and result[2] > 0.1:  # Low confidence threshold
                                    fallback_lines.append(result[1].strip())
                            
                            if fallback_lines:
                                extracted_lines = fallback_lines
                                print(f"[DEBUG] EasyOCR fallback found {len(extracted_lines)} lines for task {task_id}")
                            else:
                                extracted_lines = [f"Text extraction partially failed: {str(extraction_error)[:100]}..."]
                        else:
                            extracted_lines = [f"Text extraction failed but processing completed: {str(extraction_error)[:100]}..."]
                    else:
                        extracted_lines = [f"Could not read image file: {str(extraction_error)[:100]}..."]
                else:
                    extracted_lines = [f"Processing encountered an error: {str(extraction_error)[:100]}..."]
            except Exception as fallback_error:
                print(f"[ERROR] Fallback also failed for task {task_id}: {fallback_error}")
                extracted_lines = [f"Text extraction failed: {str(extraction_error)[:100]}..."]
            
            # Continue with PDF generation even if extraction had issues
            print(f"[DEBUG] Continuing with {len(extracted_lines)} lines despite extraction error")
        
        # Generate PDF with progress
        processing_status[task_id].update({
            'progress': 90,
            'message': 'Generating PDF document...'
        })
        
        print(f"[DEBUG] Starting PDF generation for task {task_id} with {len(extracted_lines)} lines")
        
        # Save to PDF with timeout to prevent hanging
        pdf_generation_start = time.time()
        try:
            pdf_path = pipeline.save_to_pdf(extracted_lines, file_path, RESULTS_FOLDER)
            pdf_time = time.time() - pdf_generation_start
            print(f"[DEBUG] PDF generation completed in {pdf_time:.2f} seconds for task {task_id}")
        except Exception as pdf_error:
            print(f"[ERROR] PDF generation failed for task {task_id}: {pdf_error}")
            # Create a simple fallback PDF
            pdf_path = create_simple_fallback_pdf(extracted_lines, file_path, RESULTS_FOLDER)
            print(f"[DEBUG] Fallback PDF created for task {task_id}")
        
        # Force completion to ensure it never gets stuck
        processing_status[task_id].update({
            'progress': 95,
            'message': 'Finalizing document...'
        })
        
        time.sleep(0.1)  # Brief pause
        
        processing_status[task_id].update({
            'progress': 98,
            'message': 'Almost done...'
        })
        
        # Prepare results
        result_data = {
            'lines': extracted_lines,
            'line_count': len(extracted_lines),
            'pdf_path': pdf_path,
            'processed_at': datetime.now().isoformat(),
            'warp': warp,
            'spellcheck': spellcheck
        }
        
        # Force final completion
        processing_status[task_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Processing completed successfully!',
            'result': result_data
        }
        
        print(f"[DEBUG] Task {task_id} completed successfully with {len(extracted_lines)} lines")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[ERROR] Task {task_id} failed after {elapsed_time:.2f} seconds: {e}")
        
        processing_status[task_id] = {
            'status': 'error',
            'progress': 100,  # Set to 100 so UI knows it's done
            'message': f'Error processing image: {str(e)}',
            'error': str(e)
        }
    
    # Final safety check - ensure we never leave a task in limbo
    finally:
        if task_id in processing_status:
            current_status = processing_status[task_id]
            if current_status.get('status') == 'processing' and current_status.get('progress', 0) < 100:
                # Force completion if still processing
                elapsed_time = time.time() - start_time
                print(f"[WARNING] Force completing task {task_id} after {elapsed_time:.2f} seconds")
                
                processing_status[task_id].update({
                    'status': 'completed' if current_status.get('progress', 0) > 50 else 'error',
                    'progress': 100,
                    'message': 'Processing completed (forced)' if current_status.get('progress', 0) > 50 else 'Processing failed (timeout)',
                })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    pipeline_type = request.form.get('pipeline', 'fast')  # Default to fast pipeline for speed
    warp_flag = request.form.get('warp', 'false').lower() in ['1','true','yes','on']
    spellcheck_flag = request.form.get('spellcheck', 'false').lower() in ['1','true','yes','on']
    simple_flag = request.form.get('simple', 'true').lower() in ['1','true','yes','on']
    
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
            args=(file_path, task_id, pipeline_type, warp_flag, spellcheck_flag)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'task_id': task_id,
            'message': 'File uploaded successfully. Processing started.',
            'filename': filename,
'warp': warp_flag,
            'spellcheck': spellcheck_flag,
            'simple': simple_flag
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

@app.route('/debug/status')
def debug_status():
    """Debug endpoint to see all task statuses"""
    return jsonify({
        'total_tasks': len(processing_status),
        'tasks': processing_status,
        'timestamp': datetime.now().isoformat()
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
    print("="*60)
    
    # Preload fast pipeline for instant first request
    print("Preloading Fast OCR Pipeline for instant processing...")
    try:
        pipeline_cache['fast'] = FastOCRPipeline()
        print("✓ Fast pipeline ready!")
    except Exception as e:
        print(f"⚠ Could not preload pipeline: {e}")
    
    print("="*60)
    print("Available at: http://localhost:5000")
    print("="*60)
    
    # Use a more robust server configuration
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Reduce verbose logging
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down OCR Web Application...")
    except Exception as e:
        print(f"Server error: {e}")
>>>>>>> a3deb97 (Change in OCR Accuracy)
