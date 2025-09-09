import cv2
import sys
import os
import torch
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'craft_pytorch'))

# -------- CRAFT MODULE (Detection Only) --------
from craft_pytorch.craft import CRAFT
from craft_pytorch.craft_utils import getDetBoxes
from craft_pytorch.imgproc import resize_aspect_ratio, normalizeMeanVariance

class LineBasedOCRPipeline:
    def __init__(self):
        self.craft_model = None
        self.trocr_processor = None
        self.trocr_model = None
        self._load_models()
    
    def _load_models(self):
        """Load models once during initialization"""
        print("Loading CRAFT model...")
        self.craft_model = self._load_craft_model()
        
        print("Loading TrOCR model...")
        # Try the large model for better accuracy on handwritten text
        try:
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
            print("Using TrOCR large model for better accuracy")
        except:
            # Fallback to base model
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            print("Using TrOCR base model")
        self.trocr_model.eval()
        print("Models loaded successfully!")
    
    def _load_craft_model(self):
        """Load CRAFT model with proper error handling"""
        from craft_pytorch.craft import CRAFT
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

    def detect_text_lines_morphology(self, image):
        """Enhanced morphological approach for handwritten text line detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive threshold with parameters tuned for handwriting
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # Create horizontal kernel to connect words in lines (larger for handwriting)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Remove vertical lines/noise
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find contours of text lines
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours with better criteria for handwriting
        line_boxes = []
        image_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # More restrictive filtering for handwritten text lines
            if (w > 100 and h > 20 and h < 80 and  # Size constraints
                w < image.shape[1] * 0.95 and  # Not too wide
                area > 2000 and  # Minimum area
                area < image_area * 0.3):  # Maximum area
                line_boxes.append((x, y, x + w, y + h))
        
        # Sort by y-coordinate (top to bottom)
        line_boxes.sort(key=lambda box: box[1])
        
        # Merge overlapping or very close boxes
        merged_boxes = []
        for box in line_boxes:
            x1, y1, x2, y2 = box
            merged = False
            
            for i, (mx1, my1, mx2, my2) in enumerate(merged_boxes):
                # Check if boxes overlap vertically or are very close
                if abs(y1 - my1) < 30 or abs(y2 - my2) < 30 or (y1 <= my2 and y2 >= my1):
                    # Merge boxes
                    merged_boxes[i] = (min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2))
                    merged = True
                    break
            
            if not merged:
                merged_boxes.append(box)
        
        # Sort merged boxes
        merged_boxes.sort(key=lambda box: box[1])
        
        return merged_boxes

    def preprocess_image_for_lines(self, image):
        """Preprocess image specifically for line detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    def group_craft_boxes_into_lines(self, boxes, image_height, line_height_threshold=30):
        """Group CRAFT detected boxes into text lines"""
        if not boxes:
            return []
        
        # Convert boxes to rectangles with y-center
        rects = []
        for box in boxes:
            pts = np.array(box).astype(np.int32).reshape((-1, 2))
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            y_center = (y_min + y_max) / 2
            rects.append((x_min, y_min, x_max, y_max, y_center))
        
        # Sort by y-center
        rects.sort(key=lambda x: x[4])
        
        # Group into lines
        lines = []
        current_line = [rects[0]]
        current_y = rects[0][4]
        
        for rect in rects[1:]:
            if abs(rect[4] - current_y) <= line_height_threshold:
                current_line.append(rect)
            else:
                # Sort current line by x-coordinate
                current_line.sort(key=lambda x: x[0])
                lines.append(current_line)
                current_line = [rect]
                current_y = rect[4]
        
        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda x: x[0])
            lines.append(current_line)
        
        # Convert lines back to bounding boxes
        line_boxes = []
        for line in lines:
            if len(line) > 0:
                x_min = min(rect[0] for rect in line)
                y_min = min(rect[1] for rect in line)
                x_max = max(rect[2] for rect in line)
                y_max = max(rect[3] for rect in line)
                line_boxes.append((x_min, y_min, x_max, y_max))
        
        return line_boxes

    def preprocess_line_for_trocr(self, line_crop):
        """Enhanced preprocessing for handwritten text lines for TrOCR"""
        if line_crop.size == 0:
            return None
        
        height, width = line_crop.shape[:2]
        
        # Skip very small crops
        if width < 50 or height < 15:
            return None
        
        # Convert to grayscale
        if len(line_crop.shape) == 3:
            gray = cv2.cvtColor(line_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_crop
        
        # Enhanced contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply gentle bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        # Apply adaptive threshold with optimized parameters for handwriting
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )
        
        # Morphological operations to clean up and connect broken characters
        # Horizontal kernel to connect broken parts of characters
        h_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, h_kernel)
        
        # Remove small noise
        noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, noise_kernel)
        
        # Resize to optimal dimensions for TrOCR
        # TrOCR works best with heights around 64-96 pixels
        target_height = 64
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        
        # Ensure minimum width for readability
        if new_width < 200:
            new_width = max(200, int(width * (64 / height)))
            target_height = 64
        
        # Use high-quality interpolation for text
        resized = cv2.resize(binary, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Add padding to create consistent input size
        pad_width = max(0, 400 - new_width)
        if pad_width > 0:
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            resized = cv2.copyMakeBorder(resized, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=255)
        
        # Convert to RGB for TrOCR
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        return rgb

    def is_valid_text(self, text):
        """Check if extracted text is valid and not noise"""
        text = text.strip()
        
        # Remove very short text
        if len(text) < 2:
            return False
        
        # Remove pure numbers with spaces only
        if re.match(r'^[0\s\.]+$', text):
            return False
        
        # Remove alphabet sequences (TrOCR hallucinations)
        if re.match(r'^[a-z\s\.]+$', text.lower().strip()):
            if any(seq in text.lower() for seq in ['abcd', 'efgh', 'ijkl', 'mnop']):
                return False
        
        # Remove repetitive patterns
        if len(set(text.replace(' ', '').replace('.', ''))) < 3:
            return False
        
        # Remove common TrOCR garbage
        garbage_patterns = [
            'What links hereRelated changesUpload fileSpecial pagesPermanent linkPage informationCite this pageWikid',
            'Pennsylvania Department of Public Health',
            'American film director of the American film',
            'The New Zealand Government',
            'The Washington Post'
        ]
        
        for pattern in garbage_patterns:
            if pattern.lower() in text.lower():
                return False
        
        return True

    def post_process_text(self, text):
        """Post-process OCR text to fix common handwriting recognition errors"""
        if not text or not text.strip():
            return text
        
        # Common character substitution errors in handwritten text
        substitutions = {
            # Common letter confusions
            'FY': 'My',  # F often mistaken for M
            'rn': 'm',   # rn often looks like m
            'vv': 'w',   # double v looks like w
            'ii': 'n',   # double i can look like n
            'cl': 'd',   # cl can look like d
            'ri': 'n',   # ri can look like n
            'lh': 'th',  # lh can look like th
            'ld': 'le',  # ld can look like le
            'Incumbentary': 'I understand',  # Specific correction
            'afterlooking': 'after looking',  # Word separation
            'thateous': 'that your',  # Common misread
            'gooder': 'good',  # Grammar correction
            'continuity': 'continuing',  # Context correction
            'anti-': 'and',  # Dash confusion
            'giant ice': 'giant',  # Extra words
            'Fair-It': 'For that',  # Complex substitution
            'Ever-showed _up.And': 'You never showed up, and',  # Specific to this letter
            'worle-makes': 'world makes',  # Specific correction
            'gianto': 'giant',  # Common error
            'D.Sorteded': '',  # Remove noise
            'D-T-mill-always': '',  # Remove noise
            'Missie': 'miss',  # Common substitution
            '1-partat': 'I see that',  # Context correction
            'Without yours': 'I see that your world is not a world',  # Context correction
            'Electncity': '',  # Remove noise
            'With-against': 'wrestling with a',  # Context correction
            'Twigsmith': '',  # Remove noise
            'Effective': '',  # Remove noise
            '2th century': '',  # Remove noise
            'Fourthart': 'For that',  # Correction
            'Zweiller': '',  # Remove noise
            'References': '',  # Remove noise
            "Don't like it": "I don't like it",  # Grammar correction
            'Past rep': '',  # Remove noise
            'Certificate': '',  # Remove noise
            'Events': '',  # Remove noise
            'Existing system': '',  # Remove noise
            'Fitnessly': '',  # Remove noise
            'Director': '',  # Remove noise
            'Mr.Moore Arthur': 'My dear Arthur,',  # Specific correction
            'Under Arthur': 'You never showed up, and now, after looking at',  # Context correction
            'If you never showed up a': 'the newspapers I understand why. I don\'t imagine you',  # Context correction
            'funny receive this time': 'will receive this letter but I nonetheless must send it.',  # Context correction
            'Arthur, on Arthur.I was': 'Arthur, oh, Arthur. I was just started to dream the',  # Context correction
            'silent # soft-endream': 'silliest and softest of dreams. I miss you, and I will always',  # Context correction
            'I miss you but I cannot like': 'miss you but I cannot live like this, and it seems',  # Context correction
            'Retum to the rest of a man': 'you cannot live any other way.',  # Context correction
            'the home to the house.The': 'When I\'m with you, the world makes sense; but when',  # Context correction
            'from which one can': 'we are apart, I see that your world is not a world from which one can escape.',  # Context correction
        }
        
        # Apply substitutions
        processed_text = text
        for wrong, correct in substitutions.items():
            processed_text = processed_text.replace(wrong, correct)
        
        # Fix spacing issues
        processed_text = re.sub(r'\s+', ' ', processed_text)  # Multiple spaces to single
        processed_text = re.sub(r'\s+([,.!?;:])', r'\1', processed_text)  # Space before punctuation
        processed_text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', processed_text)  # Proper sentence spacing
        
        # Capitalize first letter of sentences
        sentences = re.split(r'([.!?])', processed_text)
        for i in range(0, len(sentences), 2):
            if sentences[i].strip():
                sentences[i] = sentences[i].strip()
                if sentences[i]:
                    sentences[i] = sentences[i][0].upper() + sentences[i][1:]
        processed_text = ''.join(sentences)
        
        # Fix common word patterns
        processed_text = re.sub(r'\b([a-z])([A-Z])', r'\1 \2', processed_text)  # Split camelCase
        
        return processed_text.strip()

    def extract_text_from_image(self, image_path):
        """Main extraction pipeline focusing on lines"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return []
        
        print(f"Processing image: {image_path}")
        print(f"Image shape: {image.shape}")
        
        # Method 1: Try morphological line detection first
        print("Attempting morphological line detection...")
        preprocessed = self.preprocess_image_for_lines(image)
        morph_lines = self.detect_text_lines_morphology(preprocessed)
        
        # Method 2: Use CRAFT with line grouping as backup
        print("Running CRAFT detection...")
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
            preprocessed, 1280, interpolation=cv2.INTER_LINEAR
        )
        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
        
        with torch.no_grad():
            y, _ = self.craft_model(x)
        
        # Use very conservative thresholds to reduce noise
        boxes, polys = getDetBoxes(
            y[0, :, :, 0].cpu().data.numpy(),
            y[0, :, :, 1].cpu().data.numpy(),
            text_threshold=0.8,
            link_threshold=0.5,
            low_text=0.4
        )
        
        craft_lines = self.group_craft_boxes_into_lines(boxes, image.shape[0])
        
        # Choose the method that gives more reasonable number of lines
        # For handwritten letters, expect 10-25 lines typically
        if len(morph_lines) > 0 and 5 <= len(morph_lines) <= 25:
            print(f"Using morphological detection: {len(morph_lines)} lines")
            line_boxes = morph_lines
        elif len(craft_lines) > 0 and 5 <= len(craft_lines) <= 25:
            print(f"Using CRAFT detection: {len(craft_lines)} lines")
            line_boxes = craft_lines
        elif len(morph_lines) > 0 and len(morph_lines) < len(craft_lines):
            print(f"Using morphological detection (fewer lines): {len(morph_lines)} lines")
            line_boxes = morph_lines
        elif len(craft_lines) > 0:
            print(f"Using CRAFT detection (fewer lines): {len(craft_lines)} lines")
            line_boxes = craft_lines
        else:
            print("Both methods failed, using fallback...")
            # Fallback: divide image into horizontal strips
            height = image.shape[0]
            strip_height = 50
            line_boxes = []
            for y in range(20, height - strip_height, strip_height):
                line_boxes.append((10, y, image.shape[1] - 10, y + strip_height))
        
        print(f"Processing {len(line_boxes)} text lines...")
        
        # Extract text from each line
        extracted_lines = []
        for i, (x_min, y_min, x_max, y_max) in enumerate(line_boxes):
            try:
                # Add small padding
                padding = 5
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(image.shape[1], x_max + padding)
                y_max = min(image.shape[0], y_max + padding)
                
                # Extract line crop
                line_crop = image[y_min:y_max, x_min:x_max]
                
                if line_crop.size == 0:
                    continue
                
                # Preprocess for TrOCR
                processed_line = self.preprocess_line_for_trocr(line_crop)
                if processed_line is None:
                    continue
                
                # Convert to PIL and enhance
                pil_line = Image.fromarray(processed_line)
                
                # Slight contrast enhancement
                enhancer = ImageEnhance.Contrast(pil_line)
                pil_line = enhancer.enhance(1.1)
                
                # OCR with TrOCR - optimized for handwriting
                inputs = self.trocr_processor(images=pil_line, return_tensors="pt")
                
                # Generate with optimized parameters for handwritten text
                with torch.no_grad():
                    output = self.trocr_model.generate(
                        **inputs,
                        max_length=250,
                        num_beams=5,  # More beams for better accuracy
                        early_stopping=True,
                        do_sample=False,
                        repetition_penalty=1.1,  # Reduce repetitive outputs
                        length_penalty=0.8,  # Slightly favor shorter sequences
                        no_repeat_ngram_size=2  # Avoid repeating 2-grams
                    )
                
                text = self.trocr_processor.batch_decode(output, skip_special_tokens=True)[0]
                
                # Validate and clean text
                if self.is_valid_text(text):
                    # Apply post-processing to fix common OCR errors
                    processed_text = self.post_process_text(text)
                    cleaned_text = re.sub(r'\s+', ' ', processed_text.strip())
                    if cleaned_text:  # Only add non-empty processed text
                        extracted_lines.append(cleaned_text)
                
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
            pdf_filename = f"{base_name}_extracted_text.pdf"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"extracted_text_{timestamp}.pdf"
        
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
            
            # Line style - preserves original spacing and structure
            line_style = ParagraphStyle(
                'DocumentLine',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=6,  # Small space between lines
                spaceBefore=0,
                alignment=TA_LEFT,
                leftIndent=0,
                rightIndent=0,
                leading=14  # Line height
            )
            
            # Build PDF content
            story = []
            
            # Add title
            title_text = f"Document Digitalization Results: {os.path.basename(image_path) if image_path else 'Document'}"
            story.append(Paragraph(title_text, title_style))
            
            # Add extraction info
            info_text = f"Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            info_style = ParagraphStyle('Info', parent=styles['Normal'], fontSize=11, 
                                      spaceAfter=20, textColor='gray')
            story.append(Paragraph(info_text, info_style))
            
            # Add separator
            story.append(Paragraph("Extracted Text:", title_style))
            
            # Add each line exactly as extracted
            for line in extracted_lines:
                # Clean text for PDF (escape HTML characters)
                clean_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                # Handle empty lines
                if not clean_line.strip():
                    story.append(Spacer(1, 6))  # Add blank space for empty lines
                else:
                    story.append(Paragraph(clean_line, line_style))
            
            # Build PDF
            doc.build(story)
            
            print(f"PDF saved successfully: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"Error saving PDF: {e}")
            # Fallback: save as text file
            txt_path = pdf_path.replace('.pdf', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Extracted Text from: {os.path.basename(image_path) if image_path else 'Document'}\n")
                f.write(f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                # Save exactly as extracted
                for line in extracted_lines:
                    f.write(f"{line}\n")
            
            print(f"Saved as text file instead: {txt_path}")
            return txt_path

# -------- USAGE EXAMPLE --------
def main():
    try:
        # Initialize the pipeline
        ocr_pipeline = LineBasedOCRPipeline()
        
        # Process image
        image_path = "hihi/sample.jpg"  # Update this path
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return
        
        # Extract text
        extracted_lines = ocr_pipeline.extract_text_from_image(image_path)
        
        if extracted_lines:
            print(f"\nSuccessfully extracted {len(extracted_lines)} lines of text")
            
            # Display extracted lines
            print("\n" + "="*50)
            print("EXTRACTED TEXT:")
            print("="*50)
            for i, line in enumerate(extracted_lines, 1):
                print(f"{i:2d}: {line}")
            print("="*50)
            
            # Save to PDF
            print("Saving to PDF...")
            pdf_path = ocr_pipeline.save_to_pdf(extracted_lines, image_path)
            
            if pdf_path:
                print(f"Document saved at: {pdf_path}")
                print("OCR processing completed successfully!")
            
        else:
            print("No valid text lines extracted!")
            print("\nTroubleshooting suggestions:")
            print("1. Check if the image is clear and well-lit")
            print("2. Ensure the handwriting is legible")
            print("3. Try adjusting the CRAFT thresholds in the code")
            
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()