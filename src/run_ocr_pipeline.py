import cv2
import sys
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'craft_pytorch'))

# -------- CRAFT MODULE (Detection Only) --------
from craft_pytorch.craft import CRAFT
from craft_pytorch.craft_utils import getDetBoxes
from craft_pytorch.imgproc import resize_aspect_ratio, normalizeMeanVariance

# -------- CRAFT MODEL LOADER --------
def load_craft_model():
    from craft_pytorch.craft import CRAFT
    model = CRAFT()

    weights_path = os.path.join("craft_pytorch", "weights", "craft_mlt_25k.pth")
    state_dict = torch.load(weights_path, map_location="cpu")

    # Handle 'module.' prefix in keys
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

# -------- (NEW) PREPROCESSING FOR CROP BEFORE TrOCR --------
def resize_with_padding(image, target_size=(384, 384)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (target_size[0] - new_h) // 2
    bottom = target_size[0] - new_h - top
    left = (target_size[1] - new_w) // 2
    right = target_size[1] - new_w - left

    # Pad with white pixels
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded

def preprocess_crop(crop):
    # -------- 1️⃣ Resize and pad to 384x384 (TrOCR friendly) --------
    padded = resize_with_padding(crop, target_size=(384, 384))

    # -------- 2️⃣ Convert to grayscale --------
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)

    # -------- 3️⃣ Convert grayscale back to 3-channel RGB for TrOCR --------
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return rgb




# -------- TrOCR MODULE (Recognition) --------
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")



def recognize_with_trocr(pil_image):
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# -------- FULL COMBINED PIPELINE --------
def extract_text_from_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return []

    # Resize + normalize for CRAFT
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR)
    ratio_h = ratio_w = 1 / target_ratio
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()

    # Load models
    craft_model = load_craft_model()
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    trocr_model.eval()

    # Run CRAFT Detection
    with torch.no_grad():
        y, _ = craft_model(x)
    boxes, polys = getDetBoxes(
        y[0, :, :, 0].cpu().data.numpy(),
        y[0, :, :, 1].cpu().data.numpy(),
        text_threshold=0.7, link_threshold=0.4, low_text=0.4
    )

    texts = []
    for box in boxes:
        pts = np.array(box).astype(np.int32).reshape((-1, 2))
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        if x_max - x_min < 1 or y_max - y_min < 1:
            continue

        crop = image[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue

        # -------- 👇 Preprocess the cropped image before passing to TrOCR --------
        processed_crop = preprocess_crop(crop)  # NEW: clean, binarize, denoise
        pil_crop = Image.fromarray(processed_crop)  # Convert to PIL RGB image

        try:
            inputs = processor(images=pil_crop, return_tensors="pt")
            output = trocr_model.generate(**inputs)
            text = processor.batch_decode(output, skip_special_tokens=True)[0]
            texts.append(text)
        except:
            continue

    return texts

# -------- RUN EXAMPLE --------
if __name__ == "__main__":
    image_path = "hihi/sample.png"
    output = extract_text_from_image(image_path)
    print("📄 Final Extracted Text:\n", output)
