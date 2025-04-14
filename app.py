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

# Import our custom modules
import encryption_utils
from sr_inference import create_sr_model, process_image, process_video
from ocr_utils import extract_text_from_image, extract_text_with_regions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure constants
MODEL_DIR = "model"
KEYS_DIR = "keys"
SERVER_PRIVATE_KEY = os.path.join(KEYS_DIR, "server_private_key.pem")
CLIENT_PUBLIC_KEY = os.path.join(KEYS_DIR, "client_public_key.pem")
TEMP_DIR = tempfile.gettempdir()

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Ensure encryption keys are set up
# Replace line 44 (the encryption_utils.ensure_keys_exist line) with this error handling code:

# Add debugging information
import sys
print("Python path:", sys.path)
print("Current directory:", os.getcwd())
print("Directory contents:", os.listdir("."))
if os.path.exists("encryption_utils.py"):
    print("encryption_utils.py exists in current directory")
    
# Try to load the keys with error handling
try:
    print("Attempting to call ensure_keys_exist...")
    print("Functions in encryption_utils:", [f for f in dir(encryption_utils) if not f.startswith('_')])
    encryption_utils.ensure_keys_exist(KEYS_DIR)
    print("Keys setup successful")
except AttributeError as e:
    print(f"Function not found: {e}")
    print("Using fallback key check implementation")
    os.makedirs(KEYS_DIR, exist_ok=True)
    
    # Just check if keys exist
    server_private_key_path = os.path.join(KEYS_DIR, "server_private_key.pem")
    server_public_key_path = os.path.join(KEYS_DIR, "server_public_key.pem")
    client_private_key_path = os.path.join(KEYS_DIR, "client_private_key.pem")
    client_public_key_path = os.path.join(KEYS_DIR, "client_public_key.pem")
    
    # Log status of key files
    print(f"Server private key exists: {os.path.exists(server_private_key_path)}")
    print(f"Client public key exists: {os.path.exists(client_public_key_path)}")

# Global model instance
sr_model = None

def initialize_model():
    """
    Initialize the super-resolution model.
    """
    global sr_model
    
    # Determine model path - use RealESRGAN by default
    model_path = os.path.join(MODEL_DIR, "RealESRGAN_x4plus.pth")
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}. "
                      "Please download the model file and place it in the model directory.")
        
        # Create a placeholder model file (for development)
        with open(model_path, 'wb') as f:
            f.write(b'PLACEHOLDER_MODEL')
            
        logger.warning(f"Created a placeholder model file at {model_path}")
    
    # Create the SR model
    try:
        sr_model = create_sr_model("real-esrgan", model_path)
        logger.info("Super-resolution model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing super-resolution model: {e}")
        logger.error("Using placeholder implementation")
        
        # Use a placeholder implementation
        from sr_inference import SuperResolutionModel
        
        class PlaceholderModel(SuperResolutionModel):
            def load_model(self):
                logger.warning("Using placeholder model that simply resizes images")
                self.model = None
                
            def preprocess(self, img):
                return img
                
            def postprocess(self, output):
                return output
                
            def enhance(self, img, text_hint=None):
                # Simple 4x upscaling using OpenCV
                h, w = img.shape[:2]
                return cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
                
        sr_model = PlaceholderModel(model_path)


def process_encrypted_image(encrypted_data):
    """
    Process an encrypted image through the pipeline.
    
    Args:
        encrypted_data: Base64-encoded encrypted image data
        
    Returns:
        Dictionary with processing results and enhanced image
    """
    # Decrypt the image data
    try:
        image_bytes = encryption_utils.decrypt_data(encrypted_data, SERVER_PRIVATE_KEY)
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        return {"status": "error", "message": "Failed to decrypt image data"}
    
    # Convert bytes to numpy array
    try:
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        return {"status": "error", "message": "Failed to decode image data"}
    
    # Initial OCR
    try:
        initial_ocr = extract_text_with_regions(image)
        text_hint = initial_ocr.get('text', '')
        logger.info(f"Initial OCR result: {text_hint[:50]}...")
    except Exception as e:
        logger.error(f"Initial OCR error: {e}")
        text_hint = ""
    
    # Run super-resolution enhancement
    try:
        enhanced_image = process_image(image, sr_model, text_hint)
    except Exception as e:
        logger.error(f"Super-resolution error: {e}")
        return {"status": "error", "message": "Failed to enhance image"}
    
    # OCR on enhanced image
    try:
        final_ocr = extract_text_with_regions(enhanced_image)
        logger.info(f"Final OCR result: {final_ocr.get('text', '')[:50]}...")
    except Exception as e:
        logger.error(f"Final OCR error: {e}")
        final_ocr = {"text": "", "confidence": 0}
    
    # Encode the enhanced image
    _, buffer = cv2.imencode('.png', enhanced_image)
    enhanced_bytes = buffer.tobytes()
    
    # Prepare the result dictionary
    result = {
        "enhanced_image": enhanced_bytes,
        "original_ocr": initial_ocr,
        "enhanced_ocr": final_ocr,
        "processing_info": {
            "original_size": image.shape,
            "enhanced_size": enhanced_image.shape,
            "enhancement_factor": enhanced_image.shape[1] / image.shape[1]
        }
    }
    
    return result


def process_encrypted_video(encrypted_data):
    """
    Process an encrypted video through the pipeline.
    
    Args:
        encrypted_data: Base64-encoded encrypted video data
        
    Returns:
        Dictionary with processing results and enhanced video
    """
    # Decrypt the video data
    try:
        video_bytes = encryption_utils.decrypt_data(encrypted_data, SERVER_PRIVATE_KEY)
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        return {"status": "error", "message": "Failed to decrypt video data"}
    
    # Create temporary input and output files
    temp_input_path = os.path.join(TEMP_DIR, f"temp_input_{int(time.time())}.mp4")
    temp_output_path = os.path.join(TEMP_DIR, f"temp_output_{int(time.time())}.mp4")
    
    try:
        # Save the decrypted video to a temp file
        with open(temp_input_path, 'wb') as f:
            f.write(video_bytes)
        
        # Process the video
        processing_result = process_video(
            temp_input_path,
            temp_output_path,
            sr_model,
            ocr_on_keyframes=True,
            keyframe_interval=30
        )
        
        # Read the enhanced video into memory
        with open(temp_output_path, 'rb') as f:
            enhanced_video_bytes = f.read()
            
        # Prepare the result dictionary
        result = {
            "enhanced_video": enhanced_video_bytes,
            "ocr_results": processing_result.get('ocr_results', []),
            "processing_info": {
                "frames_processed": processing_result.get('frames_processed', 0),
                "processing_time": processing_result.get('processing_time', 0),
                "input_resolution": processing_result.get('input_resolution', (0, 0)),
                "output_resolution": processing_result.get('output_resolution', (0, 0)),
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
    Process encrypted media (image or video) and return enhanced result.
    
    Expects JSON with:
    {
        "encrypted_media": "<base64 RSA blob>",
        "media_type": "image" or "video"
    }
    
    Returns JSON with:
    {
        "status": "success" or "error",
        "encrypted_result": "<base64 RSA blob>" or
        "message": "error message"
    }
    """
    start_time = time.time()
    
    # Check for correct JSON format
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Validate required fields
    if 'encrypted_media' not in data or 'media_type' not in data:
        return jsonify({
            "status": "error", 
            "message": "Request must contain 'encrypted_media' and 'media_type'"
        }), 400
    
    encrypted_media = data['encrypted_media']
    media_type = data['media_type'].lower()
    
    # Process based on media type
    if media_type == 'image':
        try:
            result = process_encrypted_image(encrypted_media)
            
            if 'status' in result and result['status'] == 'error':
                return jsonify(result), 500
                
            # Encrypt the result
            encrypted_result = encryption_utils.encrypt_data(
                json.dumps(
                    {
                        "enhanced_image": base64.b64encode(result["enhanced_image"]).decode('utf-8'),
                        "original_ocr": result["original_ocr"],
                        "enhanced_ocr": result["enhanced_ocr"],
                        "processing_info": result["processing_info"]
                    }
                ).encode('utf-8'),
                CLIENT_PUBLIC_KEY
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Image processed successfully in {processing_time:.2f}s")
            
            return jsonify({
                "status": "success",
                "encrypted_result": encrypted_result,
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
            result = process_encrypted_video(encrypted_media)
            
            if 'status' in result and result['status'] == 'error':
                return jsonify(result), 500
                
            # Encrypt the result
            encrypted_result = encryption_utils.encrypt_data(
                json.dumps(
                    {
                        "enhanced_video": base64.b64encode(result["enhanced_video"]).decode('utf-8'),
                        "ocr_results": result["ocr_results"],
                        "processing_info": result["processing_info"]
                    }
                ).encode('utf-8'),
                CLIENT_PUBLIC_KEY
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Video processed successfully in {processing_time:.2f}s")
            
            return jsonify({
                "status": "success",
                "encrypted_result": encrypted_result,
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


@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": sr_model is not None,
        "encryption_ready": os.path.exists(SERVER_PRIVATE_KEY) and os.path.exists(CLIENT_PUBLIC_KEY)
    })


# Modify the bottom of your app.py file
if __name__ == '__main__':
    
    # Initialize the super-resolution model
    initialize_model()
    
    # Use environment variable for port if provided (for hosting platforms)
    port = int(os.environ.get('PORT', 5000))
    
    # In production, don't use debug mode
    app.run(host='0.0.0.0', port=port, debug=False)
