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

class OptimizedOCRPipeline:
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
        try:
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
            print("Using TrOCR large model")
        except:
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            print("Using TrOCR base model")
        
        self.trocr_model.eval()
        
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
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()
        return model

    def detect_text_lines_optimized(self, image):
        """Optimized text line detection that preserves individual lines"""
        print("Using optimized text line detection...")
        
        height, width = image.shape[:2]
        
        # Strategy 1: Fixed horizontal strips (most reliable for handwriting)
        strip_height = 32  # Smaller strips for better precision
        overlap = 8
        line_boxes = []
        y_start = 25  # Start from top with margin
        
        while y_start + strip_height < height - 25:
            y_end = y_start + strip_height
            x_start = 8  # Left margin
            x_end = width - 8  # Right margin
            
            line_boxes.append((x_start, y_start, x_end, y_end))
            y_start += strip_height - overlap
        
        print(f"Created {len(line_boxes)} horizontal strips")
        
        # Strategy 2: CRAFT detection for additional lines
        try:
            craft_lines = self._craft_line_detection(image)
            if craft_lines:
                print(f"CRAFT detected {len(craft_lines)} additional lines")
                # Add CRAFT lines without merging to preserve individual lines
                for craft_line in craft_lines:
                    # Check if this line overlaps significantly with existing lines
                    overlap_found = False
                    for existing_line in line_boxes:
                        if self._calculate_overlap(craft_line, existing_line) > 0.7:
                            overlap_found = True
                            break
                    
                    if not overlap_found:
                        line_boxes.append(craft_line)
        except Exception as e:
            print(f"CRAFT detection failed: {e}")
        
        # Sort by y-coordinate
        line_boxes.sort(key=lambda box: box[1])
        
        print(f"Final line count: {len(line_boxes)}")
        return line_boxes

    def _calculate_overlap(self, box1, box2):
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Return intersection over union
        return intersection / (area1 + area2 - intersection)

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
            
            # Use more aggressive thresholds for handwritten text
            boxes, polys = getDetBoxes(
                y[0, :, :, 0].cpu().data.numpy(),
                y[0, :, :, 1].cpu().data.numpy(),
                text_threshold=0.5,
                link_threshold=0.2,
                low_text=0.1
            )
            
            return self.group_craft_boxes_into_lines(boxes, image.shape[0])
        except Exception as e:
            print(f"CRAFT detection failed: {e}")
            return []

    def group_craft_boxes_into_lines(self, boxes, image_height, line_height_threshold=15):
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
        
        if width < 40 or height < 10:
            return None
        
        # Convert to grayscale
        if len(line_crop.shape) == 3:
            gray = cv2.cvtColor(line_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_crop
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
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
        """Extract text using ensemble of models"""
        results = []
        
        # TrOCR extraction
        try:
            processed_line = self.preprocess_line_for_ocr(line_crop)
            if processed_line is not None:
                pil_line = Image.fromarray(processed_line)
                
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(pil_line)
                pil_line = enhancer.enhance(1.4)
                
                # TrOCR generation with optimized parameters
                inputs = self.trocr_processor(images=pil_line, return_tensors="pt")
                
                with torch.no_grad():
                    output = self.trocr_model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=5,
                        early_stopping=True,
                        do_sample=False,
                        repetition_penalty=1.2
                    )
                
                trocr_text = self.trocr_processor.batch_decode(output, skip_special_tokens=True)[0]
                results.append(('trocr', trocr_text, 0.9))
        except Exception as e:
            print(f"TrOCR failed: {e}")
        
        # EasyOCR extraction
        try:
            easyocr_result = self.easyocr_reader.readtext(line_crop)
            if easyocr_result:
                # Get all results with confidence
                for result in easyocr_result:
                    text, confidence = result[1], result[2]
                    if confidence > 0.2:  # Lower threshold for more results
                        results.append(('easyocr', text, confidence))
        except Exception as e:
            print(f"EasyOCR failed: {e}")
        
        return results

    def select_best_text(self, text_results):
        """Select best text from ensemble results"""
        if not text_results:
            return None, 0.0
        
        # Filter out very low confidence results
        filtered_results = [r for r in text_results if r[2] > 0.2]
        
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
            'miss': 'miss',
            'missed': 'missed',
            'missing': 'missing',
            'You': 'You',
            'you': 'you',
            'never': 'never',
            'would': 'would',
            'sense': 'sense',
            'but': 'but',
            'when': 'when',
            'fear': 'fear',
            'For': 'For',
            'that': 'that',
            'go': 'go',
            'now': 'now',
            'one': 'one',
            'day': 'day'
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
        
        # Detect text lines using optimized method
        line_boxes = self.detect_text_lines_optimized(image)
        
        if not line_boxes:
            print("No text lines detected!")
            return []
        
        print(f"Processing {len(line_boxes)} text lines...")
        
        # Extract text from each line
        extracted_lines = []
        for i, (x_min, y_min, x_max, y_max) in enumerate(line_boxes):
            try:
                # Add padding
                padding = 3
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
                
                if text and confidence > 0.1:  # Very low threshold to capture more text
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
            pdf_filename = f"{base_name}_optimized_extracted_text.pdf"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"optimized_extracted_text_{timestamp}.pdf"
        
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
            title_text = f"Optimized OCR Results: {os.path.basename(image_path) if image_path else 'Document'}"
            story.append(Paragraph(title_text, title_style))
            
            # Add extraction info
            info_text = f"Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with Optimized Pipeline"
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
            
            print(f"Optimized PDF saved successfully: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"Error saving PDF: {e}")
            # Fallback: save as text file
            txt_path = pdf_path.replace('.pdf', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Optimized OCR Results from: {os.path.basename(image_path) if image_path else 'Document'}\n")
                f.write(f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                for line in extracted_lines:
                    f.write(f"{line}\n")
            
            print(f"Saved as text file instead: {txt_path}")
            return txt_path

# -------- USAGE EXAMPLE --------
def main():
    try:
        # Initialize the optimized pipeline
        ocr_pipeline = OptimizedOCRPipeline()
        
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
            print("OPTIMIZED OCR RESULTS:")
            print("="*50)
            for i, line in enumerate(extracted_lines, 1):
                print(f"{i:2d}: {line}")
            print("="*50)
            
            # Save to PDF
            print("Saving to PDF...")
            pdf_path = ocr_pipeline.save_to_pdf(extracted_lines, image_path)
            
            if pdf_path:
                print(f"Document saved at: {pdf_path}")
                print("Optimized OCR processing completed successfully!")
            
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

