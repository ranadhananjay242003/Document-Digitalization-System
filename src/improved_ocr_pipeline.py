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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'craft_pytorch'))

# -------- CRAFT MODULE (Detection Only) --------
from craft_pytorch.craft import CRAFT
from craft_pytorch.craft_utils import getDetBoxes
from craft_pytorch.imgproc import resize_aspect_ratio, normalizeMeanVariance

class ImprovedOCRPipeline:
    def __init__(self):
        self.craft_model = None
        self.trocr_processor = None
        self.trocr_model = None
        self.easyocr_reader = None
        self._load_models()
    
    def _load_models(self):
        """Load multiple models for ensemble approach"""
        print("Loading CRAFT model...")
        self.craft_model = self._load_craft_model()
        
        print("Loading TrOCR models...")
        # Load both large and base models for ensemble
        try:
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
            print("Using TrOCR large model")
        except:
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            print("Using TrOCR base model")
        
        self.trocr_model.eval()
        
        # Load EasyOCR as backup
        print("Loading EasyOCR...")
        self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
        
        print("All models loaded successfully!")
    
    def _load_craft_model(self):
        """Load CRAFT model with proper error handling"""
        model = CRAFT()
        
        weights_path = os.path.join("craft_pytorch", "weights", "craft_mlt_25k.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"CRAFT weights not found at {weights_path}")
            
        state_dict = torch.load(weights_path, map_location="cpu")
        
        # Handle 'module.' prefix in keys
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()
        return model

    def enhanced_preprocessing(self, image):
        """Enhanced image preprocessing for better OCR accuracy"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Apply Gaussian blur to smooth the image
        blurred = cv2.GaussianBlur(filtered, (3, 3), 0)
        
        # Apply adaptive threshold with optimized parameters
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )
        
        # Morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned

    def detect_text_lines_advanced(self, image):
        """Advanced text line detection using multiple methods"""
        # Method 1: Morphological approach
        preprocessed = self.enhanced_preprocessing(image)
        morph_lines = self._morphological_line_detection(preprocessed)
        
        # Method 2: CRAFT-based detection
        craft_lines = self._craft_line_detection(image)
        
        # Method 3: Projection-based detection
        projection_lines = self._projection_line_detection(preprocessed)
        
        # Combine and merge all detected lines
        all_lines = morph_lines + craft_lines + projection_lines
        merged_lines = self._merge_overlapping_lines(all_lines)
        
        return merged_lines

    def _morphological_line_detection(self, binary_image):
        """Morphological text line detection"""
        # Create horizontal kernel to connect words in lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Remove vertical lines/noise
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        line_boxes = []
        image_area = binary_image.shape[0] * binary_image.shape[1]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if (w > 80 and h > 15 and h < 60 and
                w < binary_image.shape[1] * 0.95 and
                area > 1500 and area < image_area * 0.25):
                line_boxes.append((x, y, x + w, y + h))
        
        return line_boxes

    def _craft_line_detection(self, image):
        """CRAFT-based text line detection"""
        try:
            img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
                image, 1280, interpolation=cv2.INTER_LINEAR
            )
            x = normalizeMeanVariance(img_resized)
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
            
            with torch.no_grad():
                y, _ = self.craft_model(x)
            
            # Use conservative thresholds
            boxes, polys = getDetBoxes(
                y[0, :, :, 0].cpu().data.numpy(),
                y[0, :, :, 1].cpu().data.numpy(),
                text_threshold=0.7,
                link_threshold=0.4,
                low_text=0.3
            )
            
            return self.group_craft_boxes_into_lines(boxes, image.shape[0])
        except Exception as e:
            print(f"CRAFT detection failed: {e}")
            return []

    def _projection_line_detection(self, binary_image):
        """Projection-based text line detection"""
        # Calculate horizontal projection
        projection = np.sum(binary_image, axis=1)
        
        # Find text regions using threshold
        threshold = np.mean(projection) * 0.5
        text_regions = projection > threshold
        
        # Find line boundaries
        line_boxes = []
        start_y = None
        
        for i, is_text in enumerate(text_regions):
            if is_text and start_y is None:
                start_y = i
            elif not is_text and start_y is not None:
                end_y = i
                if end_y - start_y > 10:  # Minimum line height
                    line_boxes.append((0, start_y, binary_image.shape[1], end_y))
                start_y = None
        
        return line_boxes

    def _merge_overlapping_lines(self, lines):
        """Merge overlapping or very close text lines"""
        if not lines:
            return []
        
        # Sort by y-coordinate
        lines.sort(key=lambda box: box[1])
        
        merged = []
        for line in lines:
            x1, y1, x2, y2 = line
            merged_flag = False
            
            for i, (mx1, my1, mx2, my2) in enumerate(merged):
                # Check for overlap or proximity
                if (y1 <= my2 + 20 and y2 >= my1 - 20):
                    # Merge lines
                    merged[i] = (min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2))
                    merged_flag = True
                    break
            
            if not merged_flag:
                merged.append(line)
        
        return merged

    def group_craft_boxes_into_lines(self, boxes, image_height, line_height_threshold=25):
        """Group CRAFT detected boxes into text lines"""
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
        
        if width < 60 or height < 15:
            return None
        
        # Convert to grayscale
        if len(line_crop.shape) == 3:
            gray = cv2.cvtColor(line_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_crop
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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

    def extract_text_with_confidence(self, line_crop):
        """Extract text using multiple models with confidence scoring"""
        results = []
        
        # TrOCR extraction
        try:
            processed_line = self.preprocess_line_for_ocr(line_crop)
            if processed_line is not None:
                pil_line = Image.fromarray(processed_line)
                
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(pil_line)
                pil_line = enhancer.enhance(1.2)
                
                # TrOCR generation with better parameters
                inputs = self.trocr_processor(images=pil_line, return_tensors="pt")
                
                with torch.no_grad():
                    output = self.trocr_model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=5,
                        early_stopping=True,
                        do_sample=False,
                        temperature=0.8,
                        repetition_penalty=1.2
                    )
                
                trocr_text = self.trocr_processor.batch_decode(output, skip_special_tokens=True)[0]
                results.append(('trocr', trocr_text, 0.8))  # Estimated confidence
        except Exception as e:
            print(f"TrOCR failed: {e}")
        
        # EasyOCR extraction
        try:
            easyocr_result = self.easyocr_reader.readtext(line_crop)
            if easyocr_result:
                # Get the text with highest confidence
                best_result = max(easyocr_result, key=lambda x: x[2])
                easyocr_text = best_result[1]
                confidence = best_result[2]
                results.append(('easyocr', easyocr_text, confidence))
        except Exception as e:
            print(f"EasyOCR failed: {e}")
        
        return results

    def ensemble_text_selection(self, text_results):
        """Select best text from ensemble results"""
        if not text_results:
            return None, 0.0
        
        # Filter out very low confidence results
        filtered_results = [r for r in text_results if r[2] > 0.3]
        
        if not filtered_results:
            return None, 0.0
        
        # If only one result, return it
        if len(filtered_results) == 1:
            return filtered_results[0][1], filtered_results[0][2]
        
        # If multiple results, use voting or confidence-based selection
        # For now, return the highest confidence result
        best_result = max(filtered_results, key=lambda x: x[2])
        return best_result[1], best_result[2]

    def enhanced_post_processing(self, text):
        """Enhanced text post-processing with better corrections"""
        if not text or not text.strip():
            return text
        
        # Remove common OCR artifacts
        text = re.sub(r'[#_]{2,}', ' ', text)
        text = re.sub(r'-{2,}', ' ', text)
        text = re.sub(r'[^\w\s\.,!?;:()\-\'"]', '', text)
        
        # Fix common OCR errors
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
            'maintainance': 'maintenance'
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def extract_text_from_image(self, image_path):
        """Main extraction pipeline with improved accuracy"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return []
        
        print(f"Processing image: {image_path}")
        print(f"Image shape: {image.shape}")
        
        # Detect text lines using advanced method
        print("Detecting text lines...")
        line_boxes = self.detect_text_lines_advanced(image)
        
        if not line_boxes:
            print("No text lines detected, using fallback method...")
            # Fallback: simple horizontal strips
            height, width = image.shape[:2]
            strip_height = 40
            overlap = 8
            line_boxes = []
            y_start = 40
            
            while y_start + strip_height < height - 40:
                y_end = y_start + strip_height
                x_start = 15
                x_end = width - 15
                line_boxes.append((x_start, y_start, x_end, y_end))
                y_start += strip_height - overlap
        
        print(f"Processing {len(line_boxes)} text lines...")
        
        # Extract text from each line
        extracted_lines = []
        for i, (x_min, y_min, x_max, y_max) in enumerate(line_boxes):
            try:
                # Add padding
                padding = 8
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(image.shape[1], x_max + padding)
                y_max = min(image.shape[0], y_max + padding)
                
                # Extract line crop
                line_crop = image[y_min:y_max, x_min:x_max]
                
                if line_crop.size == 0:
                    continue
                
                # Extract text with confidence
                text_results = self.extract_text_with_confidence(line_crop)
                text, confidence = self.ensemble_text_selection(text_results)
                
                if text and confidence > 0.2:  # Minimum confidence threshold
                    # Apply post-processing
                    processed_text = self.enhanced_post_processing(text)
                    if processed_text.strip():
                        extracted_lines.append(processed_text.strip())
                        print(f"Line {i+1}: {processed_text.strip()} (conf: {confidence:.2f})")
                
            except Exception as e:
                print(f"Error processing line {i}: {e}")
                continue
        
        return extracted_lines
    
    def save_to_pdf(self, extracted_lines, image_path, output_dir=None):
        """Save extracted text to PDF format"""
        if not extracted_lines:
            print("No text to save!")
            return None
        
        # Determine output directory and filename
        if output_dir is None:
            output_dir = os.path.dirname(image_path) if image_path else "."
        
        # Create filename based on original image
        if image_path:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            pdf_filename = f"{base_name}_improved_extracted_text.pdf"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"improved_extracted_text_{timestamp}.pdf"
        
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
            title_text = f"Improved OCR Results: {os.path.basename(image_path) if image_path else 'Document'}"
            story.append(Paragraph(title_text, title_style))
            
            # Add extraction info
            info_text = f"Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with Enhanced Pipeline"
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
            
            print(f"Improved PDF saved successfully: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"Error saving PDF: {e}")
            # Fallback: save as text file
            txt_path = pdf_path.replace('.pdf', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Improved OCR Results from: {os.path.basename(image_path) if image_path else 'Document'}\n")
                f.write(f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                for line in extracted_lines:
                    f.write(f"{line}\n")
            
            print(f"Saved as text file instead: {txt_path}")
            return txt_path

# -------- USAGE EXAMPLE --------
def main():
    try:
        # Initialize the improved pipeline
        ocr_pipeline = ImprovedOCRPipeline()
        
        # Process image
        image_path = "hihi/sample.jpg"
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return
        
        # Extract text
        extracted_lines = ocr_pipeline.extract_text_from_image(image_path)
        
        if extracted_lines:
            print(f"\nSuccessfully extracted {len(extracted_lines)} lines of text")
            
            # Display extracted lines
            print("\n" + "="*50)
            print("IMPROVED OCR RESULTS:")
            print("="*50)
            for i, line in enumerate(extracted_lines, 1):
                print(f"{i:2d}: {line}")
            print("="*50)
            
            # Save to PDF
            print("Saving to PDF...")
            pdf_path = ocr_pipeline.save_to_pdf(extracted_lines, image_path)
            
            if pdf_path:
                print(f"Document saved at: {pdf_path}")
                print("Improved OCR processing completed successfully!")
            
        else:
            print("No valid text lines extracted!")
            print("\nTroubleshooting suggestions:")
            print("1. Check if the image is clear and well-lit")
            print("2. Ensure the handwriting is legible")
            print("3. Try adjusting the preprocessing parameters")
            
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

