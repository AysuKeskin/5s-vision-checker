import glob
import os
import cv2
from gemini_prompt import get_object_list_from_gemini_vision
from grounded_dino_wrapper import detect_objects

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

    # Step 3: Run Grounded DINO on the image
    annotated_img, detected_labels = detect_objects(IMAGE_PATH, dynamic_object_list)
    print("✅ Detected:", detected_labels)

    # Step 4: Save and display
    base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{base}_annotated.jpg")
    cv2.imwrite(OUTPUT_PATH, annotated_img)
    print(f"💾 Output saved to {OUTPUT_PATH}")



    # Optional: Open image window
    #cv2.imshow("Detection", annotated_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
