"""
Quick test script to verify OCR text extraction improvements
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fast_ocr_pipeline import FastOCRPipeline

def test_ocr():
    print("=" * 80)
    print("Testing OCR Text Extraction")
    print("=" * 80)
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = FastOCRPipeline()
    
    # Get image path from user
    print("\n" + "=" * 80)
    image_path = input("Enter path to image file (or press Enter for default): ").strip()
    
    if not image_path:
        # Try common paths
        test_paths = [
            "hihi/sample.jpg",
            "hihi/sample.png",
            "uploads",
            "test.jpg",
            "sample.jpg"
        ]
        
        for path in test_paths:
            if os.path.exists(path):
                if os.path.isdir(path):
                    # Get first image in directory
                    for file in os.listdir(path):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(path, file)
                            break
                else:
                    image_path = path
                    break
        
        if not image_path:
            print("\nError: No image file found!")
            print("Please specify an image path.")
            return
    
    if not os.path.exists(image_path):
        print(f"\nError: Image not found at: {image_path}")
        return
    
    print(f"\nProcessing: {image_path}")
    print("=" * 80)
    
    try:
        # Extract text
        lines = pipeline.extract_text_from_image(image_path)
        
        # Display results
        print("\n" + "=" * 80)
        print("EXTRACTED TEXT:")
        print("=" * 80)
        
        for i, line in enumerate(lines, 1):
            if line.strip():
                print(f"{i:3d}. {line}")
            else:
                print()  # Paragraph break
        
        print("=" * 80)
        print(f"\nTotal lines: {len(lines)}")
        print(f"Non-empty lines: {len([l for l in lines if l.strip()])}")
        
        # Save to PDF
        print("\n" + "=" * 80)
        save = input("\nSave to PDF? (y/n): ").strip().lower()
        
        if save == 'y':
            output_folder = "test_results"
            os.makedirs(output_folder, exist_ok=True)
            
            pdf_path = pipeline.save_to_pdf(lines, image_path, output_folder)
            print(f"\nPDF saved to: {pdf_path}")
        
        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ocr()

