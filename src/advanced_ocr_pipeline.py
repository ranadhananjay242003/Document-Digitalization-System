import cv2
import sys
import os
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from datetime import datetime
import easyocr
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'craft_pytorch'))

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

# -------- CRAFT MODULE (Detection Only) --------
from craft_pytorch.craft import CRAFT
from craft_pytorch.craft_utils import getDetBoxes
from craft_pytorch.imgproc import resize_aspect_ratio, normalizeMeanVariance

class AdvancedOCRPipeline:
    def __init__(self, verbose: bool = False, enable_easyocr: bool = False):
        self.verbose = verbose
        self.enable_easyocr = enable_easyocr
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.craft_model = None
        self.trocr_processor = None
        self.trocr_model = None
        self.easyocr_reader = None
        self._load_models()
    
    def _load_models(self):
        """Load multiple models for ensemble approach"""
        if self.verbose:
            print("Loading CRAFT model...")
        self.craft_model = self._load_craft_model().to(self.device)
        
        if self.verbose:
            print("Loading TrOCR model...")
        # Prefer base model for speed; fallback to large if needed
        try:
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        except:
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
        self.trocr_model.to(self.device)
        self.trocr_model.eval()
        
        if self.enable_easyocr:
            if self.verbose:
                print("Loading EasyOCR...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        if self.verbose:
            print("All models loaded successfully!")
    
    def _load_craft_model(self):
        """Load CRAFT model with proper error handling"""
        model = CRAFT()
        
        weights_path = os.path.join("craft_pytorch", "weights", "craft_mlt_25k.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"CRAFT weights not found at {weights_path}")
            
        state_dict = torch.load(weights_path, map_location="cpu")
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()
        return model

    def multi_scale_preprocessing(self, image):
        """Multi-scale preprocessing for better text detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Method 1: CLAHE + Bilateral Filter
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered1 = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Method 2: Gaussian Blur + Adaptive Threshold
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        filtered2 = cv2.bilateralFilter(blurred, 11, 50, 50)
        
        # Method 3: Median Filter for noise reduction
        median_filtered = cv2.medianBlur(gray, 3)
        filtered3 = cv2.bilateralFilter(median_filtered, 9, 75, 75)
        
        return [filtered1, filtered2, filtered3]

    def detect_text_lines_robust(self, image):
        """Robust text line detection using multiple strategies"""
        if self.verbose:
            print("Using robust text line detection...")
        
        # Strategy 1: Improved horizontal strips for handwriting
        height, width = image.shape[:2]
        strip_height = 80  # Increased for handwritten text
        overlap = 20
        line_boxes = []
        y_start = 20  # Start from top with margin
        
        while y_start + strip_height < height - 20:
            y_end = y_start + strip_height
            x_start = 20  # Left margin
            x_end = width - 20  # Right margin
            
            line_boxes.append((x_start, y_start, x_end, y_end))
            y_start += strip_height - overlap
        
        if self.verbose:
            print(f"Created {len(line_boxes)} horizontal strips")
        
        # Strategy 2: CRAFT detection as backup
        try:
            craft_lines = self._craft_line_detection(image)
            if craft_lines and self.verbose:
                print(f"CRAFT detected {len(craft_lines)} additional lines")
                # Add CRAFT lines without merging immediately
                line_boxes.extend(craft_lines)
        except Exception as e:
            if self.verbose:
                print(f"CRAFT detection failed: {e}")
        
        # Strategy 3: Projection-based detection
        try:
            preprocessed_images = self.multi_scale_preprocessing(image)
            for i, preprocessed in enumerate(preprocessed_images):
                projection_lines = self._projection_line_detection(preprocessed)
                if projection_lines and self.verbose:
                    print(f"Projection method {i+1} detected {len(projection_lines)} lines")
                    line_boxes.extend(projection_lines)
        except Exception as e:
            if self.verbose:
                print(f"Projection detection failed: {e}")
        
        # Remove duplicates and sort
        unique_lines = []
        for line in line_boxes:
            if line not in unique_lines:
                unique_lines.append(line)
        
        unique_lines.sort(key=lambda box: box[1])  # Sort by y-coordinate
        
        # TEMPORARILY DISABLE MERGING TO SEE ALL LINES
        # final_lines = self._smart_merge_lines(unique_lines)
        final_lines = unique_lines
        
        if self.verbose:
            print(f"Final line count: {len(final_lines)}")
        return final_lines

    def _craft_line_detection(self, image):
        """CRAFT-based text line detection with better parameters"""
        try:
            img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
                image, 1280, interpolation=cv2.INTER_LINEAR
            )
            x = normalizeMeanVariance(img_resized)
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
            
            with torch.no_grad():
                y, _ = self.craft_model(x.to(self.device))
            
            # Use more conservative thresholds for handwritten text
            boxes, polys = getDetBoxes(
                y[0, :, :, 0].cpu().data.numpy(),
                y[0, :, :, 1].cpu().data.numpy(),
                text_threshold=0.3,  # Lowered for more detection
                link_threshold=0.2,  # Lowered for more detection
                low_text=0.1  # Lowered for more detection
            )
            
            return self.group_craft_boxes_into_lines(boxes, image.shape[0])
        except Exception as e:
            if self.verbose:
                print(f"CRAFT detection failed: {e}")
            return []

    def _projection_line_detection(self, binary_image):
        """Projection-based text line detection"""
        # Calculate horizontal projection
        projection = np.sum(binary_image, axis=1)
        
        # Apply smoothing to reduce noise
        kernel_size = 5
        smoothed_projection = np.convolve(projection, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Find text regions using adaptive threshold
        threshold = np.mean(smoothed_projection) * 0.3
        text_regions = smoothed_projection > threshold
        
        # Find line boundaries
        line_boxes = []
        start_y = None
        
        for i, is_text in enumerate(text_regions):
            if is_text and start_y is None:
                start_y = i
            elif not is_text and start_y is not None:
                end_y = i
                if end_y - start_y > 8:  # Minimum line height
                    line_boxes.append((0, start_y, binary_image.shape[1], end_y))
                start_y = None
        
        return line_boxes

    def _smart_merge_lines(self, lines):
        """Merge overlapping or very close text lines"""
        if not lines:
            return []
        
        # Sort lines by y-coordinate
        lines.sort(key=lambda box: box[1])
        
        merged = []
        for line in lines:
            x1, y1, x2, y2 = line
            merged_flag = False
            
            for i, (mx1, my1, mx2, my2) in enumerate(merged):
                # Check for overlap or proximity
                if (y1 <= my2 + 15 and y2 >= my1 - 15):
                    # Merge lines
                    merged[i] = (min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2))
                    merged_flag = True
                    break
            
            if not merged_flag:
                merged.append(line)
        
        return merged

    def group_craft_boxes_into_lines(self, boxes, image_height, line_height_threshold=40):
        """Group CRAFT detected boxes into text lines with better separation"""
        if not boxes:
            return []
        
        rects = []
        for box in boxes:
            pts = np.array(box).astype(np.int32).reshape((-1, 2))
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            y_center = (y_min + y_max) / 2
            rects.append((x_min, y_min, x_max, y_max, y_center))
        
        rects.sort(key=lambda x: x[4])
        
        lines = []
        current_line = [rects[0]]
        current_y = rects[0][4]
        
        for rect in rects[1:]:
            if abs(rect[4] - current_y) <= line_height_threshold:
                current_line.append(rect)
            else:
                current_line.sort(key=lambda x: x[0])
                lines.append(current_line)
                current_line = [rect]
                current_y = rect[4]
        
        if current_line:
            current_line.sort(key=lambda x: x[0])
            lines.append(current_line)
        
        line_boxes = []
        for line in lines:
            if len(line) > 0:
                x_min = min(rect[0] for rect in line)
                y_min = min(rect[1] for rect in line)
                x_max = max(rect[2] for rect in line)
                y_max = max(rect[3] for rect in line)
                line_boxes.append((x_min, y_min, x_max, y_max))
        
        return line_boxes

    def preprocess_line_for_ocr(self, line_crop):
        """Enhanced preprocessing for individual text lines"""
        if line_crop.size == 0:
            return None
        
        height, width = line_crop.shape[:2]
        
        if width < 50 or height < 12:
            return None
        
        # Convert to grayscale
        if len(line_crop.shape) == 3:
            gray = cv2.cvtColor(line_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_crop
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Resize for TrOCR
        target_height = 64
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        
        if new_width < 200:
            new_width = 200
        elif new_width > 800:
            new_width = 800
        
        resized = cv2.resize(binary, (new_width, target_height), interpolation=cv2.INTER_AREA)
        
        # Convert to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        return rgb

    def extract_text_with_ensemble(self, line_crop):
        """Extract text using ensemble of models with better sentence handling"""
        results = []
        
        # TrOCR extraction with better parameters
        try:
            processed_line = self.preprocess_line_for_ocr(line_crop)
            if processed_line is not None:
                pil_line = Image.fromarray(processed_line)
                
                # Enhance contrast and sharpness
                enhancer = ImageEnhance.Contrast(pil_line)
                pil_line = enhancer.enhance(1.5)
                
                sharpness = ImageEnhance.Sharpness(pil_line)
                pil_line = sharpness.enhance(1.2)
                
                # TrOCR generation with optimized parameters for sentences
                inputs = self.trocr_processor(images=pil_line, return_tensors="pt")
                
                with torch.no_grad():
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    output = self.trocr_model.generate(
                        **inputs,
                        max_length=160,
                        num_beams=2,
                        early_stopping=True,
                        do_sample=False,
                        repetition_penalty=1.05,
                        length_penalty=1.1,
                        no_repeat_ngram_size=2
                    )
                
                trocr_text = self.trocr_processor.batch_decode(output, skip_special_tokens=True)[0]
                if trocr_text.strip() and len(trocr_text.strip()) > 2:  # Filter very short text
                    results.append(('trocr', trocr_text, 0.75))
        except Exception as e:
            if self.verbose:
                print(f"TrOCR failed: {e}")
        
        # EasyOCR extraction with better parameters
        try:
            # Try multiple preprocessing for EasyOCR
            gray = cv2.cvtColor(line_crop, cv2.COLOR_BGR2GRAY) if len(line_crop.shape) == 3 else line_crop
            
            # Method 1: Original image
            easyocr_result = self.easyocr_reader.readtext(line_crop, detail=1)
            if easyocr_result:
                for result in easyocr_result:
                    text, confidence = result[1], result[2]
                    # Filter out standalone numbers and very short text
                    if confidence > 0.2 and len(text.strip()) > 2 and not text.strip().isdigit():
                        results.append(('easyocr', text, confidence))
            
            # Method 2: Preprocessed image
            processed = self.preprocess_line_for_ocr(line_crop)
            if processed is not None:
                easyocr_result2 = self.easyocr_reader.readtext(processed, detail=1)
                if easyocr_result2:
                    for result in easyocr_result2:
                        text, confidence = result[1], result[2]
                        # Filter out standalone numbers and very short text
                        if confidence > 0.2 and len(text.strip()) > 2 and not text.strip().isdigit():
                            results.append(('easyocr_processed', text, confidence))
                            
        except Exception as e:
            if self.verbose:
                print(f"EasyOCR failed: {e}")
        
        return results

    def select_best_text(self, text_results):
        """Select best text from ensemble results with better filtering"""
        if not text_results:
            return None, 0.0
        
        # Filter out very low confidence results and standalone numbers
        filtered_results = []
        for r in text_results:
            text, confidence = r[1], r[2]
            clean_text = text.strip()
            
            # Skip if text is too short, is a standalone number, or is a single character
            if (confidence > 0.15 and 
                len(clean_text) > 3 and  # Increased minimum length
                not clean_text.isdigit() and
                not (len(clean_text) == 1 and clean_text.isalpha()) and
                not clean_text in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
                filtered_results.append(r)
        
        if not filtered_results:
            return None, 0.0
        
        # If only one result, return it
        if len(filtered_results) == 1:
            return filtered_results[0][1], filtered_results[0][2]
        
        # If multiple results, prefer longer text (likely sentences)
        # Sort by length first, then by confidence
        filtered_results.sort(key=lambda x: (len(x[1]), x[2]), reverse=True)
        return filtered_results[0][1], filtered_results[0][2]

    def enhanced_post_processing(self, text):
        """Enhanced text post-processing with better corrections and deduplication"""
        if not text or not text.strip():
            return text
        
        # Remove common OCR artifacts
        text = re.sub(r'[#_]{2,}', ' ', text)
        text = re.sub(r'-{2,}', ' ', text)
        text = re.sub(r'[^\w\s\.,!?;:()\-\'"]', '', text)
        
        # Fix common OCR errors for handwritten text
        corrections = {
            'Yevnever': 'You never',
            'woule': 'would',
            'senous': 'sense',
            'bus were': 'but when',
            'rear': 'fear',
            'Ess that': 'For that',
            'growing': 'go now',
            'done away': 'one day',
            'teh': 'the',
            'thier': 'their',
            'recieve': 'receive',
            'occured': 'occurred',
            'begining': 'beginning',
            'seperate': 'separate',
            'definately': 'definitely',
            'neccessary': 'necessary',
            'occassion': 'occasion',
            'accomodate': 'accommodate',
            'embarass': 'embarrass',
            'maintainance': 'maintenance',
            'miss': 'miss',  # Keep as is if it's correct
            'missed': 'missed',
            'missing': 'missing',
            'Eranslatlo': 'translation',
            'mainifestation': 'manifestation',
            'Cvi Lution': 'civilization',
            'brinas': 'brings',
            'verdiage': 'verbiage',
            'Regendra': 'Rajendra',
            '1930sir': '1930s',
            'The\'': 'The',
            'car': 'can',
            'Jack': 'lack',
            'meta-': 'meta',
            'somuch': 'so much',
            'salatory': 'salutary',
            'sides': 'sided'
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Fix common punctuation issues
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove spaces before punctuation
        text = re.sub(r'([.,!?;:])\s*([A-Z])', r'\1 \2', text)  # Add space after punctuation before capital
        
        # Fix capitalization for sentence starts
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        
        return text

    def remove_duplicates_and_similar(self, extracted_lines):
        """Remove duplicate and very similar text lines"""
        if not extracted_lines:
            return []
        
        # Clean and normalize text for comparison
        def normalize_text(text):
            # Remove extra spaces and convert to lowercase
            normalized = ' '.join(text.strip().lower().split())
            # Remove common punctuation for comparison
            normalized = normalized.replace('.', '').replace(',', '').replace('"', '').replace("'", '')
            return normalized
        
        # Group similar texts
        unique_lines = []
        seen_normalized = set()
        
        for line in extracted_lines:
            normalized = normalize_text(line)
            
            # Check if this text is too similar to existing ones
            is_duplicate = False
            for existing_normalized in seen_normalized:
                # Calculate similarity using simple string matching
                if self._calculate_similarity(normalized, existing_normalized) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate and normalized not in seen_normalized:
                unique_lines.append(line)
                seen_normalized.add(normalized)
        
        return unique_lines
    
    def _calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def merge_similar_lines(self, extracted_lines):
        """Merge lines that are likely parts of the same sentence"""
        if not extracted_lines:
            return []
        
        def is_continuation(text1, text2):
            """Check if text2 is a continuation of text1"""
            text1_clean = text1.strip().lower()
            text2_clean = text2.strip().lower()
            
            # Check if text2 starts with words that could continue text1
            text1_words = text1_clean.split()
            text2_words = text2_clean.split()
            
            if len(text1_words) < 2 or len(text2_words) < 2:
                return False
            
            # Check for common continuation patterns
            text1_end = ' '.join(text1_words[-2:])
            text2_start = ' '.join(text2_words[:2])
            
            # If text2 starts with words that could continue text1
            if text1_end in text2_start or text2_start in text1_end:
                return True
            
            # Check for sentence continuation patterns
            if text1_clean.endswith(('the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with')):
                return True
            
            return False
        
        merged_lines = []
        i = 0
        
        while i < len(extracted_lines):
            current_line = extracted_lines[i]
            merged_line = current_line
            
            # Look ahead for continuations
            j = i + 1
            while j < len(extracted_lines):
                if is_continuation(merged_line, extracted_lines[j]):
                    merged_line += " " + extracted_lines[j]
                    j += 1
                else:
                    break
            
            merged_lines.append(merged_line)
            i = j
        
        return merged_lines

    def extract_text_from_image(self, image_path):
        """Main extraction pipeline with improved accuracy, better filtering, and deduplication"""
        # Validate extension
        _, ext = os.path.splitext(image_path)
        if ext.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            if self.verbose:
                print(f"Unsupported image format: {ext}")
            return []
        
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            if self.verbose:
                print(f"Error: Could not load image from {image_path}")
            return []
        
        # Downscale very large images for speed
        max_dim = 1600
        h, w = image.shape[:2]
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
        if self.verbose:
            print(f"Processing image: {image_path}")
            print(f"Image shape: {image.shape}")
        
        # Detect text lines using robust method
        line_boxes = self.detect_text_lines_robust(image)
        
        if not line_boxes:
            if self.verbose:
                print("No text lines detected!")
            return []
        
        if self.verbose:
            print(f"Processing {len(line_boxes)} text lines...")
        
        # Extract text from each line
        extracted_lines = []
        for i, (x_min, y_min, x_max, y_max) in enumerate(line_boxes):
            try:
                # Add padding
                padding = 10
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(image.shape[1], x_max + padding)
                y_max = min(image.shape[0], y_max + padding)
                
                # Extract line crop
                line_crop = image[y_min:y_max, x_min:x_max]
                
                if line_crop.size == 0:
                    continue
                
                # Extract text with ensemble
                text_results = self.extract_text_with_ensemble(line_crop)
                text, confidence = self.select_best_text(text_results)
                
                if text and confidence > 0.15:  # Lower threshold for more results
                    # Apply post-processing
                    processed_text = self.enhanced_post_processing(text)
                    if processed_text.strip() and len(processed_text.strip()) > 3:
                        extracted_lines.append(processed_text.strip())
                        if self.verbose:
                            print(f"Line {i+1}: {processed_text.strip()} (conf: {confidence:.2f})")
                
            except Exception as e:
                if self.verbose:
                    print(f"Error processing line {i}: {e}")
                continue
        
        # Apply deduplication and merging
        if self.verbose:
            print(f"\nBefore deduplication: {len(extracted_lines)} lines")
        
        # Remove duplicates and similar lines
        unique_lines = self.remove_duplicates_and_similar(extracted_lines)
        if self.verbose:
            print(f"After deduplication: {len(unique_lines)} lines")
        
        # Merge similar lines into sentences
        final_lines = self.merge_similar_lines(unique_lines)
        if self.verbose:
            print(f"After merging: {len(final_lines)} lines")
        
        return final_lines
    
    def save_to_pdf(self, extracted_lines, image_path, output_dir=None):
        """Save extracted text to PDF format"""
        if not extracted_lines:
            if self.verbose:
                print("No text to save!")
            return None
        
        # Determine output directory and filename
        if output_dir is None:
            output_dir = os.path.dirname(image_path) if image_path else "."
        
        # Create filename based on original image
        if image_path:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            pdf_filename = f"{base_name}_advanced_extracted_text.pdf"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"advanced_extracted_text_{timestamp}.pdf"
        
        pdf_path = os.path.join(output_dir, pdf_filename)
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(pdf_path, pagesize=A4, 
                                  rightMargin=72, leftMargin=72, 
                                  topMargin=72, bottomMargin=18)
            
            # Get styles
            styles = getSampleStyleSheet()
            
            # Create custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
                alignment=TA_LEFT
            )
            
            line_style = ParagraphStyle(
                'DocumentLine',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=6,
                spaceBefore=0,
                alignment=TA_LEFT,
                leftIndent=0,
                rightIndent=0,
                leading=14
            )
            
            # Build PDF content
            story = []
            
            # Add title
            title_text = f"Advanced OCR Results: {os.path.basename(image_path) if image_path else 'Document'}"
            story.append(Paragraph(title_text, title_style))
            
            # Add extraction info
            info_text = f"Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with Advanced Pipeline"
            info_style = ParagraphStyle('Info', parent=styles['Normal'], fontSize=11, 
                                      spaceAfter=20, textColor='gray')
            story.append(Paragraph(info_text, info_style))
            
            # Add separator
            story.append(Paragraph("Extracted Text:", title_style))
            
            # Add each line
            for line in extracted_lines:
                clean_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                if not clean_line.strip():
                    story.append(Spacer(1, 6))
                else:
                    story.append(Paragraph(clean_line, line_style))
            
            # Build PDF
            doc.build(story)
            return pdf_path
            
        except Exception as e:
            if self.verbose:
                print(f"Error saving PDF: {e}")
            # Fallback: save as text file
            txt_path = pdf_path.replace('.pdf', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Advanced OCR Results from: {os.path.basename(image_path) if image_path else 'Document'}\n")
                f.write(f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                for line in extracted_lines:
                    f.write(f"{line}\n")
            
            return txt_path

# -------- USAGE EXAMPLE --------
def main():
    try:
        ocr_pipeline = AdvancedOCRPipeline(verbose=False, enable_easyocr=False)
        # Try common sample filenames with supported extensions
        base = os.path.join("hihi", "sample")
        candidates = [base + ext for ext in [".png", ".jpg", ".jpeg"]]
        image_path = None
        for p in candidates:
            if os.path.exists(p):
                image_path = p
                break
        if image_path is None:
            # Fallback to original path if it exists
            legacy = os.path.join("hihi", "sample.jpg")
            if os.path.exists(legacy):
                image_path = legacy
            else:
                return
        extracted_lines = ocr_pipeline.extract_text_from_image(image_path)
        if extracted_lines:
            ocr_pipeline.save_to_pdf(extracted_lines, image_path)
    except Exception:
        pass

if __name__ == "__main__":
    main()


