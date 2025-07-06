from google import genai # This is the correct import for google-genai
from google.genai import types # type: ignore # Keep this for GenerateContentConfig
from PIL import Image # For loading the image
import json # To parse JSON output from Gemini
import os # For managing API key (best practice)

# --- Gemini API Setup ---
# It's best practice to load API keys from environment variables.
# genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
# For now, using your direct key, but highly recommend environment variables for production:
client = genai.Client(api_key="AIzaSyCxWzufYMPlRAfeDZIVMLR_Zc4inoLpowQ")

# Make sure you are using a Gemini model capable of vision input
# 'gemini-pro-vision' is a good choice for this, or 'gemini-1.5-flash-001'
GEMINI_VISION_MODEL_NAME = "gemini-1.5-flash" 
# or "gemini-1.5-flash-001" if you have access and prefer it.

def get_object_list_from_gemini_vision(image_path: str) -> list:
    """
    Sends an image to Gemini Vision API and asks it to list distinct objects.
    """
    try:
        # Load the image using PIL (Pillow)
        img = Image.open(image_path).convert("RGB")

        # Prompt for Gemini Vision model
        # Ask for JSON output for easy parsing
        # Be explicit about what kind of objects you want
        prompt_content = [
            """
            Analyze this image of an indoor environment (likely an office, room, or workspace).
            Your task is to identify and list all distinct and tangible objects.
            
            **Focus Areas:**
            - Furniture (e.g., desk, chair, bookshelf, cabinet)
            - Electronics (e.g., monitor, laptop, keyboard, mouse, printer, lamp)
            - Tools and  Office Supplies (e.g., screwdriver, pen, notebook, papers, stapler, scissors)
            - Decorative and Personal Items (e.g., plant, picture frame, clock, vase, bottle, cup, flowers, book, glasses, bag)
            - Storage and Organization (e.g., box, basket, drawer, shelf)
            - Cables and Wires

            **Instructions for the List:**
            1.  Each item in the list should be a common, recognizable noun.
            2.  Use singular form (e.g., "book" not "books"). 
            3.  Prioritize specific names where ambiguity might arise (e.g., "desk lamp" instead of just "lamp" if it's clear).
            4.  Avoid adjectives (e.g., "small desk", just "desk") unless crucial for distinction.
            5.  Do not include abstract concepts, actions, or background elements like "wall", "floor", "ceiling", "light" (unless it's a specific fixture like a "ceiling light fixture").
            6.  Ensure no punctuation within the object names themselves.
            7.  Provide the output as a JSON array of strings, without any additional text or formatting outside the JSON block.

            Example JSON output (note the variety and specificity):
            ```json
            ["office desk", "office chair", "computer monitor", "keyboard", "mouse", "laptop", "desk lamp", "bookshelf", "book", "plant", "picture frame", "pen holder", "pen", "notebook", "papers", "screwdriver", "cable", "power strip", "coffee mug", "water bottle", "speaker", "external hard drive", "backpack", "waste bin"]
            ```
            """,
            img # Pass the image directly as part of the content
        ]
        gen_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
        response = client.models.generate_content(
            model=GEMINI_VISION_MODEL_NAME, 
            contents=prompt_content,
            config=gen_config # Pass the config object here
        )
        
        # Extract and parse the JSON. Gemini might wrap it in markdown.
        raw_output = response.text.strip()
        if raw_output.startswith("```json") and raw_output.endswith("```"):
            json_string = raw_output[len("```json"): -len("```")].strip()
        else:
            json_string = raw_output # Assume it's just the JSON string

        object_list = json.loads(json_string)
        # Ensure all items are strings and remove any extra whitespace/punctuation
        return [str(item).strip().lower().replace('.', '') for item in object_list]

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from Gemini: {e}. Raw response: {response.text}")
        # Fallback to a default list if parsing fails
        return ['office desk', 'tools', 'papers', 'wires', 'screwdriver', 'table', 'lamp', 'flowers', 'books']
    except Exception as e:
        print(f"An unexpected error occurred with Gemini API: {e}")
        # Fallback for any other API errors
        return ['office desk', 'tools', 'papers', 'wires', 'screwdriver', 'table', 'lamp', 'flowers', 'books']


