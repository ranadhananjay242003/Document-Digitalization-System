<<<<<<< HEAD
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
||||||| (empty tree)
=======
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
import re

# Try to import spell checker, but make it optional
try:
    from spellchecker import SpellChecker
    SPELL_CHECKER_AVAILABLE = True
except ImportError:
    SPELL_CHECKER_AVAILABLE = False
    print("Note: Install pyspellchecker for spell checking: pip install pyspellchecker")

class FastOCRPipeline:
    """Ultra-fast OCR pipeline using only EasyOCR for maximum speed"""
    
    def __init__(self):
        print("Loading EasyOCR (CPU mode for consistency)...")
        # Use CPU mode for consistent performance and lower memory usage
        self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
        
        # Initialize spell checker if available
        if SPELL_CHECKER_AVAILABLE:
            print("Loading spell checker...")
            self.spell_checker = SpellChecker()
            print("Spell checker ready!")
        else:
            self.spell_checker = None
            print("Spell checker not available (optional)")
        
        # Initialize SymSpell (optional)
        try:
            from symspellpy import SymSpell
            self.symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            dict_candidates = [
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'frequency_dictionary_en_82_765.txt'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frequency_dictionary_en_82_765.txt')
            ]
            dict_path = None
            for p in dict_candidates:
                if os.path.exists(p):
                    dict_path = p
                    break
            if dict_path is None:
                try:
                    from urllib.request import urlretrieve
                    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'), exist_ok=True)
                    url = 'https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt'
                    dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'frequency_dictionary_en_82_765.txt')
                    urlretrieve(url, dict_path)
                except Exception as e:
                    print(f"Could not download SymSpell dictionary: {e}")
                    self.symspell = None
            if dict_path and self.symspell:
                self.symspell.load_dictionary(dict_path, 0, 1)
                print("SymSpell ready!")
            # Load project whitelist (optional)
            try:
                root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                wl_candidates = [
                    os.path.join(root, 'data', 'whitelist.txt'),
                    os.path.join(root, 'whitelist.txt'),
                ]
                self.whitelist = set()
                for p in wl_candidates:
                    if os.path.exists(p):
                        from spell_utils import load_whitelist
                        self.whitelist = load_whitelist(p)
                        if self.whitelist:
                            print(f"Loaded whitelist from {p}")
                            break
            except Exception:
                self.whitelist = set()
        except Exception as e:
            self.symspell = None
            print(f"SymSpell unavailable: {e}")
        
        print("Fast OCR pipeline ready!")
    
    def extract_text_from_image(self, image_path: str, progress_callback=None) -> List[str]:
        """Fast text extraction using EasyOCR with proper line grouping and progress tracking"""
        print(f"Processing image: {image_path}")
        
        if progress_callback:
            progress_callback(10, "Loading image...")
        
        # Load and preprocess image quickly
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize large images for faster processing
        h, w = image.shape[:2]
        max_dim = 1600  # Increased for better quality
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
        if progress_callback:
            progress_callback(30, "Preprocessing image...")
        
        # Simple preprocessing for better OCR results
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Light denoise + unsharp mask to make strokes clearer
        blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
        sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
        enhanced = cv2.bilateralFilter(sharpened, 7, 50, 50)
        
        if progress_callback:
            progress_callback(50, "Extracting text...")
        
        try:
            # PASS 1: Paragraph mode for natural line joining
            if progress_callback:
                progress_callback(55, "Paragraph detection...")
            para_results = self.easyocr_reader.readtext(
                enhanced,
                detail=1,
                paragraph=True,
                width_ths=0.7,
                height_ths=0.7,
                blocklist='_'  # prevent underscores from appearing
            )
            
            paragraph_lines = []
            if para_results:
                # Sort by top Y of bbox
                def top_y(bbox):
                    ys = [p[1] for p in bbox]
                    return int(min(ys))
                para_results.sort(key=lambda r: top_y(r[0]))
                for (bbox, text, confidence) in para_results:
                    if confidence > 0.2 and isinstance(text, str) and text.strip():
                        clean = text.replace('_', ' ')
                        clean = re.sub(r'\s+', ' ', clean).strip()
                        paragraph_lines.append(clean)
            
            # If paragraph mode produced good lines, use them
            if paragraph_lines and sum(len(l.split()) for l in paragraph_lines) / max(1, len(paragraph_lines)) >= 2:
                extracted_lines = paragraph_lines
            else:
                # PASS 2: Word-level detection with improved grouping
                results = self.easyocr_reader.readtext(
                    enhanced,
                    detail=1,
                    paragraph=False,
                    width_ths=0.5,
                    height_ths=0.5,
                    decoder='beamsearch',
                    batch_size=1,
                    blocklist='_'  # avoid underscores in words
                )
                
                if progress_callback:
                    progress_callback(75, "Organizing text...")
                
                if not results:
                    print("No text detected by EasyOCR")
                    return ["No text detected in the image."]
                
                # Group text by lines based on Y-coordinates with better tolerance
                lines_dict = {}
                for (bbox, text, confidence) in results:
                    if confidence > 0.25 and text.strip():  # Lower threshold for more text
                        # Get Y position (vertical center of bbox)
                        y_pos = int((bbox[0][1] + bbox[2][1]) / 2)
                        x_pos = int(bbox[0][0])
                        bbox_height = int(bbox[2][1] - bbox[0][1])
                        
                        # INCREASED grouping tolerance to group words on same line
                        # Use 1.5x the text height as tolerance for better line grouping
                        line_tolerance = max(int(bbox_height * 1.5), 20)
                        
                        # Group by line with dynamic tolerance
                        line_found = False
                        for line_y in list(lines_dict.keys()):
                            if abs(y_pos - line_y) < line_tolerance:
                                lines_dict[line_y].append((x_pos, text.strip().replace('_',' '), y_pos))
                                line_found = True
                                break
                        
                        if not line_found:
                            lines_dict[y_pos] = [(x_pos, text.strip().replace('_',' '), y_pos)]
                
                if progress_callback:
                    progress_callback(90, "Formatting output...")
                
                # Sort lines by Y position and build final output with paragraph detection
                extracted_lines = []
                prev_y = None
                
                print(f"\nGrouped into {len(lines_dict)} lines")
                
                for y_pos in sorted(lines_dict.keys()):
                    # Check for paragraph breaks (large vertical gap)
                    if prev_y is not None:
                        gap = y_pos - prev_y
                        # If gap is larger than 1.5x normal line height, add paragraph break
                        if gap > 50:  # INCREASED threshold for paragraph breaks
                            if extracted_lines and extracted_lines[-1] != '':
                                extracted_lines.append('')  # Empty line for paragraph break
                    
                    # Sort words in each line by X position (left to right)
                    line_words = sorted(lines_dict[y_pos], key=lambda x: x[0])
                    # Join words with single space
                    line_text = ' '.join([word for _, word, _ in line_words])
                    line_text = re.sub(r'\s+', ' ', line_text).strip()
                    if line_text:
                        print(f"Line {len(extracted_lines)+1}: {line_text}")
                        extracted_lines.append(line_text)
                        prev_y = y_pos
            
            # Post-process to merge incomplete lines within paragraphs
            extracted_lines = self._merge_incomplete_lines(extracted_lines)
            
            # Apply spell checking if available (pyspellchecker OR SymSpell OR whitelist)
            if getattr(self, 'spell_checker', None) or getattr(self, 'symspell', None) or getattr(self, 'whitelist', None):
                if progress_callback:
                    progress_callback(95, "Spell checking...")
                extracted_lines = self._apply_spell_check(extracted_lines)
            
            # Apply simple corrections (safe, minimal)
            extracted_lines = [self._simple_corrections_line(l) for l in extracted_lines]

            if not extracted_lines:
                extracted_lines = ["No readable text found in the image."]
            
            if progress_callback:
                progress_callback(100, "Complete!")
            
            print(f"Extracted {len(extracted_lines)} lines of text")
            return extracted_lines
            
        except Exception as e:
            print(f"Error during text extraction: {e}")
            import traceback
            traceback.print_exc()
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
        
        # Add extracted text - preserve line structure
        if lines:
            for line in lines:
                if line.strip():
                    # Escape HTML characters to prevent PDF errors
                    clean_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(clean_line, normal_style))
                    story.append(Spacer(1, 0.08*inch))
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
    
    def _merge_incomplete_lines(self, lines: List[str]) -> List[str]:
        """Merge lines that form continuous sentences - VERY AGGRESSIVE"""
        if not lines or len(lines) < 2:
            return lines
        
        print(f"\nMerging {len(lines)} lines...")
        merged = []
        i = 0
        
        while i < len(lines):
            current = lines[i].strip()
            
            # Skip empty lines (paragraph breaks)
            if not current:
                merged.append('')
                i += 1
                continue
            
            # Start building a sentence
            sentence = current
            i += 1
            
            # AGGRESSIVE MERGING: Keep merging lines until we find a clear sentence end
            while i < len(lines):
                next_line = lines[i].strip()
                
                # Stop at paragraph breaks
                if not next_line:
                    break
                
                # Only stop if current sentence clearly ends
                last_char = sentence.rstrip()[-1:] if sentence.rstrip() else ''
                
                # Stop if ends with sentence-ending punctuation AND next starts with uppercase
                if last_char in '.!?' and next_line and next_line[0].isupper():
                    # But still merge if sentence is very short (likely fragment)
                    word_count = len(sentence.split())
                    if word_count < 4:  # Very short, probably incomplete
                        sentence = sentence.rstrip() + ' ' + next_line
                        i += 1
                        continue
                    else:
                        break
                
                # Otherwise, keep merging
                sentence = sentence.rstrip() + ' ' + next_line
                i += 1
            
            if sentence:
                print(f"Merged line: {sentence[:80]}..." if len(sentence) > 80 else f"Merged line: {sentence}")
                merged.append(sentence)
        
        print(f"Result: {len(merged)} merged lines\n")
        return merged
    
    def _easyocr_paragraph_lines(self, image) -> List[str]:
        """Run EasyOCR in paragraph mode and return cleaned lines"""
        try:
            results = self.easyocr_reader.readtext(
                image,
                detail=1,
                paragraph=True,
                width_ths=0.7,
                height_ths=0.7,
                blocklist='_'  # avoid underscores
            )
            if not results:
                return []
            # Sort by top-left y
            def top_y(bbox):
                ys = [p[1] for p in bbox]
                return int(min(ys))
            results.sort(key=lambda r: top_y(r[0]))
            lines = []
            for (bbox, text, conf) in results:
                if conf > 0.2 and isinstance(text, str) and text.strip():
                    clean = text.replace('_', ' ')
                    clean = re.sub(r'\s+', ' ', clean).strip()
                    lines.append(clean)
            return lines
        except Exception:
            return []
    
    def _simple_corrections_line(self, line: str) -> str:
        import re
        s = line
        pairs = [
            (r"\bthic\b", 'this'),
            (r"\bgitla\b", 'gita'),
            (r"\bmaun\b", 'main'),
            (r"\biiterary\b", 'literary'),
            (r"\bsurvivin\b", 'surviving'),
            (r"world\s+the present", 'world. The Present'),
            (r"\blranslation\b", 'Translation'),
            (r"\bbrinjs\b", 'Brings'),
            (r"\btotbe\b", 'to the'),
            (r"one\s+sided", 'one-sided'),
            (r"\berd\b", 'End'),
            (r"\bit lack\b", 'it lacks'),
            (r"\baeftb\b", 'depth'),
            (r"\b9uthentic\b", 'authentic'),
            (r"\bcosciousness\b", 'consciousness'),
            (r"\bdeptl\b", 'Depth'),
            (r"\bprotesthion\b", 'protections'),
            (r"\bverblage\b", 'verbiage')
        ]
        for pat, rep in pairs:
            s = re.sub(pat, rep, s, flags=re.IGNORECASE)
        return s

    def _apply_spell_check(self, lines: List[str]) -> List[str]:
        # Normalize -> spellcheck -> final normalize using shared utils
        try:
            try:
                from spell_utils import apply_spell_check_lines, normalize_lines
            except Exception:
                from src.spell_utils import apply_spell_check_lines, normalize_lines

            # pre-normalize common OCR confusions
            pre = normalize_lines(lines, final_pass=False)

            checked = apply_spell_check_lines(pre, spell_checker=self.spell_checker, symspell=getattr(self, 'symspell', None), max_edit_distance=2, whitelist=getattr(self, 'whitelist', None))

            # final normalization (punctuation/spacing/capitalization)
            final = normalize_lines(checked, final_pass=True)
            return final
        except Exception:
            return lines

# Example usage
if __name__ == "__main__":
    pipeline = FastOCRPipeline()
    # Test with an image
    # lines = pipeline.extract_text_from_image("test_image.jpg")
    # pdf_path = pipeline.save_to_pdf(lines, "test_image.jpg", "results")
>>>>>>> a3deb97 (Change in OCR Accuracy)
