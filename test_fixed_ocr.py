import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.fast_ocr_pipeline import FastOCRPipeline

def main():
    print("=" * 60)
    print("Testing Fixed OCR Pipeline")
    print("=" * 60)
    
    # Initialize the fast pipeline
    print("\n[1] Initializing Fast OCR Pipeline...")
    pipeline = FastOCRPipeline()
    
    # Test image path
    image_path = "hihi/sample.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"\n[2] Processing image: {image_path}")
    print("-" * 60)
    
    # Extract text
    extracted_lines = pipeline.extract_text_from_image(image_path)
    
    print(f"\n[3] Extracted {len(extracted_lines)} lines:")
    print("=" * 60)
    
    # Display each line with line numbers
    for i, line in enumerate(extracted_lines, 1):
        print(f"Line {i:2d}: {line}")
    
    print("=" * 60)
    
    # Generate PDF
    print(f"\n[4] Generating PDF...")
    pdf_path = pipeline.save_to_pdf(extracted_lines, image_path, "results")
    print(f"✓ PDF saved to: {pdf_path}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

