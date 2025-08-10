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
        self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
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
        """Alternative approach: Use morphological operations to detect text lines"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
        )
        
        # Create horizontal kernel to connect words in lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Create vertical kernel to separate lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        vertical_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find contours of text lines
        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours
        line_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out very small or very large regions
            if w > 50 and h > 15 and h < 100 and w < image.shape[1] * 0.9:
                line_boxes.append((x, y, x + w, y + h))
        
        # Sort by y-coordinate (top to bottom)
        line_boxes.sort(key=lambda box: box[1])
        
        return line_boxes

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
        """Preprocess a full text line for TrOCR"""
        if line_crop.size == 0:
            return None
        
        height, width = line_crop.shape[:2]
        
        # Skip very small crops
        if width < 100 or height < 20:
            return None
        
        # Convert to grayscale
        if len(line_crop.shape) == 3:
            gray = cv2.cvtColor(line_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_crop
        
        # Apply Gaussian blur to smooth the text
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # Apply adaptive threshold for clean binarization
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply light morphological operations to connect broken characters
        kernel = np.ones((1, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Resize to optimal height for TrOCR (around 64-128 pixels)
        target_height = 80
        scale = target_height / height
        new_width = int(width * scale)
        new_height = target_height
        
        resized = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
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
        if len(morph_lines) > 0 and len(morph_lines) < 50:
            print(f"Using morphological detection: {len(morph_lines)} lines")
            line_boxes = morph_lines
        elif len(craft_lines) > 0 and len(craft_lines) < 50:
            print(f"Using CRAFT detection: {len(craft_lines)} lines")
            line_boxes = craft_lines
        else:
            print("Both methods failed, using fallback...")
            # Fallback: divide image into horizontal strips
            height = image.shape[0]
            strip_height = 60
            line_boxes = []
            for y in range(0, height - strip_height, strip_height):
                line_boxes.append((0, y, image.shape[1], y + strip_height))
        
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
                
                # OCR with TrOCR
                inputs = self.trocr_processor(images=pil_line, return_tensors="pt")
                
                # Generate with more conservative parameters
                with torch.no_grad():
                    output = self.trocr_model.generate(
                        **inputs,
                        max_length=200,
                        num_beams=3,
                        early_stopping=True,
                        do_sample=False
                    )
                
                text = self.trocr_processor.batch_decode(output, skip_special_tokens=True)[0]
                
                # Validate and clean text
                if self.is_valid_text(text):
                    cleaned_text = re.sub(r'\s+', ' ', text.strip())
                    extracted_lines.append(cleaned_text)
                    # Removed: print(f"Line {i+1}: '{cleaned_text}'")  # No terminal output
                
            except Exception as e:
                print(f"Error processing line {i}: {e}")
                continue
        
        return extracted_lines
    
    def save_to_pdf(self, extracted_lines, image_path, output_dir=None):
        """Save extracted text to PDF format - only complete document with original line structure"""
        if not extracted_lines:
            print("❌ No text to save!")
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
            title_text = f"Extracted Text from: {os.path.basename(image_path) if image_path else 'Document'}"
            story.append(Paragraph(title_text, title_style))
            
            # Add extraction info
            info_text = f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            info_style = ParagraphStyle('Info', parent=styles['Normal'], fontSize=9, 
                                      spaceAfter=30, textColor='gray')
            story.append(Paragraph(info_text, info_style))
            
            # Add each line exactly as extracted (no numbering, no paragraph joining)
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
            
            print(f"✅ PDF saved successfully: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"❌ Error saving PDF: {e}")
            # Fallback: save as text file
            txt_path = pdf_path.replace('.pdf', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Extracted Text from: {os.path.basename(image_path) if image_path else 'Document'}\n")
                f.write(f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                # Save exactly as extracted
                for line in extracted_lines:
                    f.write(f"{line}\n")
            
            print(f"✅ Saved as text file instead: {txt_path}")
            return txt_path

# -------- USAGE EXAMPLE --------
def main():
    try:
        # Initialize the pipeline
        ocr_pipeline = LineBasedOCRPipeline()
        
        # Process image
        image_path = "hihi/sample.png"  # Update this path
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return
        
        extracted_lines = ocr_pipeline.extract_text_from_image(image_path)
        
        if extracted_lines:
            print(f"\n✅ Successfully extracted {len(extracted_lines)} lines of text")
            
            # Save to PDF directly without showing content
            print("💾 Saving to PDF...")
            pdf_path = ocr_pipeline.save_to_pdf(extracted_lines, image_path)
            
            if pdf_path:
                print(f"📄 Document saved at: {pdf_path}")
                print("🎉 OCR processing completed successfully!")
            
        else:
            print("❌ No valid text lines extracted!")
            print("\nTroubleshooting suggestions:")
            print("1. Check if the image is clear and well-lit")
            print("2. Ensure the handwriting is legible")
            print("3. Try adjusting the CRAFT thresholds in the code")
            
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()