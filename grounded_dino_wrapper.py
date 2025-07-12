import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import numpy as np
import cv2

# ——— Model setup ———
MODEL_ID = "IDEA-Research/grounding-dino-base"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
model     = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)

def detect_objects(image_path: str, object_list: list,
                   box_threshold=0.33, text_threshold=0.3):
    """
    image_path: path to your input image
    object_list: list of strings (e.g. ["screwdriver","notebook",...])
    Returns: (annotated_bgr_image, detected_labels)
    """
    # Load and prepare image
    image = Image.open(image_path).convert("RGB")

    # Build a single text prompt
    text_queries = " ".join([f"{obj.lower()}." for obj in object_list])

    # Inference
    inputs = processor(images=image, text=text_queries, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to get boxes/scores/labels
    target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes
    )

    # Draw boxes on the image
    img_bgr  = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    detected = set()
    for res in results:
        boxes      = res["boxes"].cpu().numpy().astype(int)
        scores     = res["scores"].cpu().numpy()
        text_labels = res["text_labels"]
        for i in range(len(boxes)):
            if scores[i] < box_threshold:
                continue
            x0, y0, x1, y1 = boxes[i]
            label = text_labels[i]
            detected.add(label)
            cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (0,255,0), 2)
            cv2.putText(
                img_bgr,
                f"{label} {scores[i]:.2f}",
                (x0, y0-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                1
            )

    return img_bgr, list(detected)
