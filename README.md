# 5S Vision Checker

A computer vision application that uses Google Gemini AI and Grounded DINO to analyze workspace images and identify objects for 5S workplace organization methodology.

## Features

- **Object Detection**: Uses Google Gemini Vision API to identify objects in workspace images
- **Zero-Shot Detection**: Implements Grounded DINO for precise object detection without training
- **5S Analysis**: Helps identify items that need sorting, organizing, or cleaning
- **Batch Processing**: Can process multiple images from input directory
- **Visual Output**: Generates annotated images showing detected objects with bounding boxes

## Prerequisites

- Python 3.10
- Google Gemini API key
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd 5S-Vision-Checker
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**
   - Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Add your API key to `gemini_prompt.py` (line 11) or set as environment variable:
     ```bash
     export GEMINI_API_KEY="your-api-key-here"
     ```

## Usage

1. **Prepare your images**
   - Place workspace images in the `inputs/` directory
   - Supported formats: JPG, PNG, JPEG

2. **Run the application**
   ```bash
   python main.py
   ```

3. **View results**
   - Annotated images will be saved in the `output/` directory
   - Object lists will be printed to console

## Project Structure

```
5S-Vision-Checker/
├── main.py                 # Main application entry point
├── gemini_prompt.py        # Google Gemini AI integration
├── grounded_dino_wrapper.py # Grounded DINO object detection
├── requirements.txt         # Python dependencies
├── inputs/                 # Input images directory
├── output/                 # Output images directory
└── venv/                   # Virtual environment (not in git)
```

## Dependencies

### Core Dependencies
- `google-genai==1.24.0` - Google Gemini AI API
- `Pillow==11.2.1` - Image processing
- `opencv-python==4.11.0.86` - Computer vision
- `torch==2.7.1` - PyTorch for deep learning
- `torchvision==0.22.1` - Computer vision models
- `transformers==4.52.4` - Hugging Face transformers (for Grounded DINO)
- `supervision==0.25.1` - Object detection utilities
- `matplotlib==3.10.3` - Plotting and visualization
- `requests==2.32.4` - HTTP requests

### Additional Dependencies
The project uses Grounded DINO for zero-shot object detection, which requires additional dependencies that are automatically installed.

## Configuration

### API Key Setup
The application uses Google Gemini AI for object detection. You need to:

1. Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add it to `gemini_prompt.py`:
   ```python
   client = genai.Client(api_key="your-api-key-here")
   ```

### Model Configuration
- **Gemini Model**: `gemini-1.5-flash` (configurable in `gemini_prompt.py`)
- **Grounded DINO**: Uses `IDEA-Research/grounding-dino-base` model

## How It Works

1. **Image Analysis**: Google Gemini Vision API analyzes the image and generates a list of objects
2. **Object Detection**: Grounded DINO uses the object list to detect and localize objects in the image
3. **Visualization**: Detected objects are highlighted with bounding boxes and labels
4. **Output**: Annotated images are saved with detection results

## Output

The application generates:
- **Annotated Images**: Images with bounding boxes and labels showing detected objects
- **Object Lists**: Text lists of detected objects
- **Console Output**: Processing status and results

## Troubleshooting

### Common Issues

1. **Import Error with Google GenAI**
   ```bash
   pip install google-genai==1.24.0
   ```

2. **CUDA/GPU Issues**
   - Install PyTorch with CUDA support if needed
   - Check GPU compatibility

3. **API Key Issues**
   - Verify your API key is correct
   - Check API quota and billing

### Environment Issues
- Ensure you're using the correct Python version (3.10+)
- Activate the virtual environment before running
- Install dependencies in the virtual environment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Acknowledgments

- [Google Gemini AI](https://ai.google.dev/) for vision capabilities
- [Grounded DINO](https://github.com/IDEA-Research/Grounded-DINO) for zero-shot object detection
- [Hugging Face Transformers](https://huggingface.co/) for model loading
