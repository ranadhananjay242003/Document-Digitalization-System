import cv2
import os
import numpy as np
from PIL import Image
import easyocr
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from datetime import datetime
from typing import List, Tuple, Optional

class FastOCRPipeline:
    """Ultra-fast OCR pipeline using only EasyOCR for maximum speed"""
    
    def __init__(self):
        print("Loading EasyOCR (CPU mode for consistency)...")
        # Use CPU mode for consistent performance and lower memory usage
        self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
        print("Fast OCR pipeline ready!")
    
    def extract_text_from_image(self, image_path: str) -> List[str]:
        """Fast text extraction using EasyOCR only"""
        print(f"Processing image: {image_path}")
        
        # Load and preprocess image quickly
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Simple preprocessing for better OCR results
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply simple enhancement
        enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        
        try:
            # Use EasyOCR with optimized settings for speed
            results = self.easyocr_reader.readtext(
                enhanced,
                detail=1,  # Get bounding boxes and confidence
                paragraph=False,  # Don't group into paragraphs for speed
                width_ths=0.7,  # Slightly relaxed for speed
                height_ths=0.7,  # Slightly relaxed for speed
                decoder='greedy'  # Faster decoding method
            )
            
            if not results:
                print("No text detected by EasyOCR")
                return ["No text detected in the image."]
            
            # Sort results by vertical position (top to bottom)
            results.sort(key=lambda x: x[0][0][1])  # Sort by top-left y coordinate
            
            # Extract text lines
            extracted_lines = []
            for (bbox, text, confidence) in results:
                # Only include text with reasonable confidence
                if confidence > 0.3 and text.strip():
                    cleaned_text = text.strip()
                    if len(cleaned_text) > 0:
                        extracted_lines.append(cleaned_text)
            
            if not extracted_lines:
                extracted_lines = ["No readable text found in the image."]
            
            print(f"Extracted {len(extracted_lines)} lines of text")
            return extracted_lines
            
        except Exception as e:
            print(f"Error during text extraction: {e}")
            return [f"Error extracting text: {str(e)}"]
    
    def save_to_pdf(self, lines: List[str], original_image_path: str, output_folder: str) -> str:
        """Save extracted text to PDF with minimal formatting for speed"""
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(original_image_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"fast_ocr_{base_name}_{timestamp}.pdf"
        output_path = os.path.join(output_folder, output_filename)
        
        # Create PDF document quickly
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        
        # Simple styles for speed
        styles = getSampleStyleSheet()
        normal_style = styles['Normal']
        
        # Build content quickly
        story = []
        
        # Add header
        header_text = f"Fast OCR Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(header_text, styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        # Add extracted text
        if lines:
            for i, line in enumerate(lines, 1):
                if line.strip():
                    # Simple paragraph for each line
                    story.append(Paragraph(f"{line}", normal_style))
                    story.append(Spacer(1, 0.1*inch))
        else:
            story.append(Paragraph("No text was extracted from the image.", normal_style))
        
        # Build PDF
        try:
            doc.build(story)
            print(f"PDF saved successfully: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error creating PDF: {e}")
            # Return a simple text file as fallback
            txt_path = output_path.replace('.pdf', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Fast OCR Results - {datetime.now()}\n")
                f.write("=" * 50 + "\n\n")
                for line in lines:
                    f.write(line + "\n")
            return txt_path

# Example usage
if __name__ == "__main__":
    pipeline = FastOCRPipeline()
    # Test with an image
    # lines = pipeline.extract_text_from_image("test_image.jpg")
    # pdf_path = pipeline.save_to_pdf(lines, "test_image.jpg", "results")
