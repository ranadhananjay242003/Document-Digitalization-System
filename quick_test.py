#!/usr/bin/env python3
"""
Quick test to verify OCR pipeline performance
"""

import os
import sys
import time

# Add src directory to path
sys.path.insert(0, 'src')

def test_pipeline():
    """Test the OCR pipeline quickly"""
    
    from src.fast_ocr_pipeline import FastOCRPipeline
    
    # Look for the most recent uploaded image
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        print("No uploads directory found - upload an image through the web interface first")
        return False
    
    # Find the most recent image file
    image_files = []
    for filename in os.listdir(uploads_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(uploads_dir, filename)
            image_files.append((filepath, os.path.getmtime(filepath)))
    
    if not image_files:
        print("No image files found in uploads directory")
        return False
    
    # Use the most recent image
    image_files.sort(key=lambda x: x[1], reverse=True)
    test_image = image_files[0][0]
    
    print(f"Testing Fast OCR Pipeline with: {os.path.basename(test_image)}")
    
    start_time = time.time()
    
    # Initialize pipeline
    pipeline = FastOCRPipeline()
    init_time = time.time() - start_time
    print(f"Pipeline initialized in {init_time:.2f}s")
    
    # Extract text
    extraction_start = time.time()
    try:
        extracted_lines = pipeline.extract_text_from_image(test_image)
        extraction_time = time.time() - extraction_start
        
        print(f"Text extracted in {extraction_time:.2f}s")
        print(f"Found {len(extracted_lines)} lines:")
        
        # Show first few lines
        for i, line in enumerate(extracted_lines[:5], 1):
            print(f"  {i}: {line}")
        
        if len(extracted_lines) > 5:
            print(f"  ... and {len(extracted_lines) - 5} more lines")
        
        # Generate PDF
        pdf_start = time.time()
        pdf_path = pipeline.save_to_pdf(extracted_lines, test_image, "results")
        pdf_time = time.time() - pdf_start
        
        print(f"PDF created in {pdf_time:.2f}s: {os.path.basename(pdf_path)}")
        
        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Quick OCR Performance Test")
    print("-" * 30)
    
    success = test_pipeline()
    
    if success:
        print("✓ Test completed successfully!")
    else:
        print("✗ Test failed!")
