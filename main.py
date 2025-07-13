import glob
import os
import cv2
from gemini_prompt import get_object_list_from_gemini_vision
from grounded_dino_wrapper import detect_objects

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# CLIP zero-shot setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def classify_organization(image_path: str) -> tuple[str, float]:
    """
    Uses multiple positive vs. negative prompts and averages
    their cosine similarities to decide.
    Returns (status, confidence).
    """
    # 1) Load & preprocess the image
    img = Image.open(image_path).convert("RGB")
    img_inputs = clip_processor(images=img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        img_feat = clip_model.get_image_features(**img_inputs)
    img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)  # normalize

    # 2) Define ensembles of prompts
    positive = [
    "a clean desk with clear surfaces",
    "office supplies neatly arranged", 
    "paperwork organized in folders",
    "minimal items on desk surface",
    "items neatly arranged in their proper places",
    "books organized on shelves"
    ]
    negative = [
        "papers scattered across desk",
        "books piled everywhere", 
        "desk surface covered with items",
        "cluttered workspace with many objects",
        "items randomly thrown about",
        "no clear system for object placement"
    ]
    # 3) Encode all text prompts
    pos_inputs = clip_processor(text=positive, return_tensors="pt", padding=True).to(DEVICE)
    neg_inputs = clip_processor(text=negative, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        pos_emb = clip_model.get_text_features(**pos_inputs)
        neg_emb = clip_model.get_text_features(**neg_inputs)

    # 4) Normalize text embeddings
    pos_emb = pos_emb / pos_emb.norm(p=2, dim=-1, keepdim=True)
    neg_emb = neg_emb / neg_emb.norm(p=2, dim=-1, keepdim=True)

    # 5) Compute cosine sims and average
    #    img_feat: (1,D), pos_emb: (3,D), neg_emb: (3,D)
    p_sims = (img_feat @ pos_emb.T).squeeze(0)  # shape (3,)
    n_sims = (img_feat @ neg_emb.T).squeeze(0)  # shape (3,)
    p_sim  = p_sims.mean().item()
    n_sim  = n_sims.mean().item()

    # 6) Decide and compute confidence margin
    status = "organized" if p_sim > n_sim else "cluttered"
    confidence = abs(p_sim - n_sim)

    return status, confidence

def add_text_with_background(img, text, position, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Text boyutlarını hesapla
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    
    # Arka plan dikdörtgeni koordinatları
    bg_x1, bg_y1 = x - 5, y - text_height - 5
    bg_x2, bg_y2 = x + text_width + 5, y + baseline + 5
    
    # Yarı şeffaf arka plan için overlay oluştur
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    # Yarı şeffaf efekt uygula
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Text'i ekle
    cv2.putText(img, text, position, font, font_scale, text_color, thickness)
    return  img
    

# ---------- Settings ----------
# Folders
INPUT_DIR  = "inputs/Office genel"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = glob.glob(os.path.join(INPUT_DIR, "*.jpg")) \
            + glob.glob(os.path.join(INPUT_DIR, "*.png"))

for IMAGE_PATH in image_paths:
    print(f"Attempting to get object list from Gemini for image: {IMAGE_PATH}")
        
    # Get the object list dynamically from Gemini's vision model
    dynamic_object_list = get_object_list_from_gemini_vision(IMAGE_PATH)
    print("Gemini-generated object list:", dynamic_object_list)

    prompt_text = ", ".join(dynamic_object_list)

    # Run Grounded DINO on the image
    annotated_img, detected_labels = detect_objects(IMAGE_PATH, dynamic_object_list)
    print("✅ Detected:", detected_labels)

    # Classify overall organization with CLIP
    status = classify_organization(IMAGE_PATH)
    print(f"→ Organization status: {status}")
    add_text_with_background(
    annotated_img, 
    f"Status: {status[0]}", 
    (10, 30),
    text_color=(255, 255, 255),  # Beyaz text
    bg_color=(0, 0, 0),          # Siyah arka plan
    alpha=0.6                    # %60 şeffaflık
    )

    # Save
    base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{base}_annotated.jpg")
    cv2.imwrite(OUTPUT_PATH, annotated_img)
    print(f"💾 Output saved to {OUTPUT_PATH}")




    # Optional: Open image window
    #cv2.imshow("Detection", annotated_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
