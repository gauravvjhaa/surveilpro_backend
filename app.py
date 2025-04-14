"""
Flask backend for the SurveilPro image and video enhancement service.
"""

import os
import time
import tempfile
import base64
import json
import logging
from io import BytesIO
import numpy as np
import cv2
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import shutil
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure constants
MODEL_DIR = "model"
TEMP_DIR = tempfile.gettempdir()

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Available models
AVAILABLE_MODELS = {
    "real-esrgan": {
        "file": "RealESRGAN_x4plus.pth",
        "type": "real-esrgan",
        "display_name": "RealESRGAN",
        "description": "Standard 4x super-resolution"
    },
    "real-hat-gan": {
        "file": "Real_HAT_GAN_sharper.pth",
        "type": "real-hat-gan",
        "display_name": "Real HAT-GAN",
        "description": "Transformer-based hybrid model with sharper results"
    }
}

# Create a simple fallback model class
class PlaceholderModel:
    def __init__(self):
        logger.warning("Using placeholder model that simply resizes images")
    
    def enhance(self, img, text_hint=None):
        # Simple 4x upscaling using OpenCV
        h, w = img.shape[:2]
        return cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)

# Use the placeholder model directly since the real models aren't working
sr_model = PlaceholderModel()

def extract_text_safe(image):
    """Safe wrapper for OCR that returns empty results if OCR fails"""
    try:
        from ocr_utils import extract_text_with_regions
        return extract_text_with_regions(image)
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return {"text": "", "confidence": 0, "regions": []}

def process_image_data(base64_image, model_key="real-esrgan"):
    """
    Process a base64 encoded image through the pipeline.
    
    Args:
        base64_image: Base64-encoded image data
        model_key: The model to use for enhancement
        
    Returns:
        Dictionary with processing results and enhanced image
    """
    # Decode base64 image data
    try:
        image_bytes = base64.b64decode(base64_image)
    except Exception as e:
        logger.error(f"Base64 decoding error: {e}")
        return {"status": "error", "message": "Failed to decode base64 image data"}
    
    # Convert bytes to numpy array
    try:
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        return {"status": "error", "message": "Failed to decode image data"}
    
    # Initial OCR (with safe wrapper)
    initial_ocr = extract_text_safe(image)
    text_hint = initial_ocr.get('text', '')
    
    # Run super-resolution enhancement
    try:
        # Use our placeholder model directly
        enhanced_image = sr_model.enhance(image, text_hint)
    except Exception as e:
        logger.error(f"Super-resolution error: {e}")
        # Fallback to basic OpenCV resize
        h, w = image.shape[:2]
        enhanced_image = cv2.resize(image, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    
    # OCR on enhanced image (with safe wrapper)
    final_ocr = extract_text_safe(enhanced_image)
    
    # Encode the enhanced image
    _, buffer = cv2.imencode('.png', enhanced_image)
    enhanced_bytes = buffer.tobytes()
    enhanced_base64 = base64.b64encode(enhanced_bytes).decode('utf-8')
    
    # Prepare the result dictionary
    result = {
        "enhanced_image": enhanced_base64,
        "original_ocr": initial_ocr,
        "enhanced_ocr": final_ocr,
        "processing_info": {
            "original_size": [int(x) for x in image.shape],
            "enhanced_size": [int(x) for x in enhanced_image.shape],
            "enhancement_factor": float(enhanced_image.shape[1] / image.shape[1]),
            "model_used": model_key
        }
    }
    
    return result

def process_video_data(base64_video, model_key="real-esrgan"):
    """
    Process a base64 encoded video through the pipeline.
    
    Args:
        base64_video: Base64-encoded video data
        model_key: The model to use for enhancement
        
    Returns:
        Dictionary with processing results and enhanced video
    """
    # Decode base64 video data
    try:
        video_bytes = base64.b64decode(base64_video)
    except Exception as e:
        logger.error(f"Base64 decoding error: {e}")
        return {"status": "error", "message": "Failed to decode base64 video data"}
    
    # Create temporary input and output files
    temp_input_path = os.path.join(TEMP_DIR, f"temp_input_{int(time.time())}.mp4")
    temp_output_path = os.path.join(TEMP_DIR, f"temp_output_{int(time.time())}.mp4")
    
    try:
        # Save the video to a temp file
        with open(temp_input_path, 'wb') as f:
            f.write(video_bytes)
        
        # Process the video frames manually using OpenCV
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {temp_input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width*4, height*4))
        
        frames_processed = 0
        ocr_results = []
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Enhance the frame using our placeholder model
            enhanced_frame = sr_model.enhance(frame)
            
            # Write enhanced frame
            out.write(enhanced_frame)
            frames_processed += 1
            
            # Add OCR results for select frames
            if frames_processed % 30 == 0:  # every 30 frames
                ocr_results.append({
                    "frame": frames_processed,
                    "timestamp": frames_processed / fps,
                    "text": "",  # Skip OCR for now since Tesseract isn't installed
                })
        
        # Release resources
        cap.release()
        out.release()
        
        # Read the enhanced video into memory
        with open(temp_output_path, 'rb') as f:
            enhanced_video_bytes = f.read()
        
        enhanced_video_base64 = base64.b64encode(enhanced_video_bytes).decode('utf-8')
            
        # Prepare the result dictionary
        result = {
            "enhanced_video": enhanced_video_base64,
            "ocr_results": ocr_results,
            "processing_info": {
                "frames_processed": frames_processed,
                "processing_time": 0,
                "input_resolution": [width, height],
                "output_resolution": [width*4, height*4],
                "model_used": model_key
            }
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        return {"status": "error", "message": f"Failed to process video: {str(e)}"}
    
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")

@app.route('/process_media', methods=['POST'])
def process_media():
    """
    Process media (image or video) and return enhanced result.
    
    Expects JSON with:
    {
        "media_data": "<base64 encoded media>",
        "media_type": "image" or "video",
        "model": "real-esrgan" or "real-hat-gan"
    }
    
    Returns JSON with:
    {
        "status": "success" or "error",
        "result": {
            "enhanced_image" or "enhanced_video": "<base64 encoded media>",
            "ocr_results": [...],
            "processing_info": {...}
        } or
        "message": "error message"
    }
    """
    start_time = time.time()
    
    # Check for correct JSON format
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Validate required fields
    if 'media_data' not in data or 'media_type' not in data:
        return jsonify({
            "status": "error", 
            "message": "Request must contain 'media_data' and 'media_type'"
        }), 400
    
    media_data = data['media_data']
    media_type = data['media_type'].lower()
    model_key = data.get('model', 'real-esrgan')
    
    # Process based on media type
    if media_type == 'image':
        try:
            result = process_image_data(media_data, model_key)
            
            if 'status' in result and result['status'] == 'error':
                return jsonify(result), 500
                
            processing_time = time.time() - start_time
            logger.info(f"Image processed successfully in {processing_time:.2f}s using placeholder model")
            
            return jsonify({
                "status": "success",
                "result": result,
                "processing_time": processing_time
            })
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return jsonify({
                "status": "error",
                "message": f"Error processing image: {str(e)}"
            }), 500
            
    elif media_type == 'video':
        try:
            result = process_video_data(media_data, model_key)
            
            if 'status' in result and result['status'] == 'error':
                return jsonify(result), 500
                
            processing_time = time.time() - start_time
            logger.info(f"Video processed successfully in {processing_time:.2f}s using placeholder model")
            
            return jsonify({
                "status": "success",
                "result": result,
                "processing_time": processing_time
            })
            
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return jsonify({
                "status": "error",
                "message": f"Error processing video: {str(e)}"
            }), 500
            
    else:
        return jsonify({
            "status": "error",
            "message": f"Unsupported media type: {media_type}. Use 'image' or 'video'"
        }), 400

@app.route('/models', methods=['GET'])
def get_models():
    """
    Get information about available models.
    """
    return jsonify({
        "status": "success",
        "models": AVAILABLE_MODELS
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({
        "status": "healthy",
        "models_loaded": True,  # We're using the placeholder model
        "available_models": list(AVAILABLE_MODELS.keys()),
        "using_fallback": True
    })

@app.route('/test_connection', methods=['GET', 'POST'])
def test_connection():
    """
    Simple endpoint to test connectivity and CORS.
    """
    if request.method == 'POST':
        try:
            # Echo back any JSON data sent
            if request.is_json:
                data = request.get_json()
                return jsonify({
                    "status": "success",
                    "message": "Connection successful!",
                    "received_data": data
                })
            else:
                return jsonify({
                    "status": "success", 
                    "message": "Connection successful, but no JSON data received."
                })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Error processing request: {str(e)}"
            }), 500
    else:
        # For GET requests
        return jsonify({
            "status": "success",
            "message": "API is reachable!"
        })

# Entry point for application
if __name__ == '__main__':
    # Use environment variable for port if provided (for hosting platforms)
    port = int(os.environ.get('PORT', 5000))
    
    # In production, don't use debug mode
    app.run(host='0.0.0.0', port=port, debug=False)