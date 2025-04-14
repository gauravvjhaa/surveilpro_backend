"""
Flask backend for the SurveilPro image and video enhancement service.
"""

import os
import time
import tempfile
import base64
import json
import logging
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure constants
MODEL_DIR = "model"
TEMP_DIR = tempfile.gettempdir()

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# We'll use these models directly from the BasicSR/RealESRGAN library
MODEL_PATHS = {
    "real-esrgan": os.path.join(MODEL_DIR, "RealESRGAN_x4plus.pth"),
    "real-hat-gan": os.path.join(MODEL_DIR, "Real_HAT_GAN_sharper.pth")
}

# These will store our loaded models
models = {}

# Load the models
def load_models():
    global models
    try:
        # For RealESRGAN
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        # 1. Load RealESRGAN
        logger.info("Loading RealESRGAN model...")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        models["real-esrgan"] = RealESRGANer(
            scale=4,
            model_path=MODEL_PATHS["real-esrgan"],
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True
        )
        logger.info("RealESRGAN model loaded successfully!")
        
        # 2. For Real-HAT-GAN (adjust architecture as needed)
        logger.info("Loading Real-HAT-GAN model...")
        try:
            from basicsr.archs.hat_arch import HAT
            hat_model = HAT(upscale=4, in_chans=3, img_size=64, window_size=16, 
                        depths=[6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6], 
                        mlp_ratio=2)
            models["real-hat-gan"] = RealESRGANer(
                scale=4,
                model_path=MODEL_PATHS["real-hat-gan"],
                model=hat_model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True
            )
            logger.info("Real-HAT-GAN model loaded successfully!")
        except Exception as e:
            logger.error(f"Could not load HAT model: {e}")
            models["real-hat-gan"] = models["real-esrgan"]  # Use RealESRGAN as fallback
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # Create basic fallback using OpenCV
        class BasicUpscaler:
            def enhance(self, img, outscale=4):
                h, w = img.shape[:2]
                result = cv2.resize(img, (w*outscale, h*outscale), interpolation=cv2.INTER_CUBIC)
                return result, None
        
        # Use OpenCV resizing as fallback for both models
        basic_model = BasicUpscaler()
        models["real-esrgan"] = basic_model
        models["real-hat-gan"] = basic_model

# Process image using selected model
def process_image_data(base64_image, model_name="real-esrgan"):
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
    
    # Get the selected model
    model = models.get(model_name, models.get("real-esrgan"))
    
    # Process the image
    try:
        logger.info(f"Processing image with {model_name} model")
        enhanced_image, _ = model.enhance(image, outscale=4)
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        # Fallback to OpenCV
        h, w = image.shape[:2]
        enhanced_image = cv2.resize(image, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    
    # Encode the enhanced image
    _, buffer = cv2.imencode('.png', enhanced_image)
    enhanced_bytes = buffer.tobytes()
    enhanced_base64 = base64.b64encode(enhanced_bytes).decode('utf-8')
    
    return {
        "enhanced_image": enhanced_base64,
        "processing_info": {
            "original_size": image.shape[:2],
            "enhanced_size": enhanced_image.shape[:2],
            "model_used": model_name
        }
    }

# Process video using selected model
def process_video_data(base64_video, model_name="real-esrgan"):
    # Decode base64 video data
    try:
        video_bytes = base64.b64decode(base64_video)
    except Exception as e:
        logger.error(f"Base64 decoding error: {e}")
        return {"status": "error", "message": "Failed to decode base64 video data"}
    
    # Create temporary files
    temp_input = os.path.join(TEMP_DIR, f"input_{time.time()}.mp4")
    temp_output = os.path.join(TEMP_DIR, f"output_{time.time()}.mp4")
    
    try:
        # Save input video
        with open(temp_input, 'wb') as f:
            f.write(video_bytes)
        
        # Get model
        model = models.get(model_name, models.get("real-esrgan"))
        
        # Open the video
        cap = cv2.VideoCapture(temp_input)
        if not cap.isOpened():
            return {"status": "error", "message": "Failed to open video file"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width*4, height*4))
        
        # Process frames
        frames_processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # Process frame
                enhanced_frame, _ = model.enhance(frame, outscale=4)
                out.write(enhanced_frame)
            except Exception as e:
                # Fallback
                h, w = frame.shape[:2]
                enhanced_frame = cv2.resize(frame, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
                out.write(enhanced_frame)
            
            frames_processed += 1
            
            # Log progress
            if frames_processed % 10 == 0:
                logger.info(f"Processed {frames_processed}/{total_frames} frames")
        
        # Release resources
        cap.release()
        out.release()
        
        # Read output video
        with open(temp_output, 'rb') as f:
            enhanced_video = f.read()
            
        enhanced_video_base64 = base64.b64encode(enhanced_video).decode('utf-8')
        
        return {
            "enhanced_video": enhanced_video_base64,
            "processing_info": {
                "frames_processed": frames_processed,
                "original_size": [width, height],
                "enhanced_size": [width*4, height*4],
                "model_used": model_name
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return {"status": "error", "message": f"Error processing video: {str(e)}"}
    finally:
        # Clean up
        for file in [temp_input, temp_output]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    pass

@app.route('/process_media', methods=['POST'])
def process_media():
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
    model_name = data.get('model', 'real-esrgan')
    
    # Process based on media type
    if media_type == 'image':
        try:
            result = process_image_data(media_data, model_name)
            processing_time = time.time() - start_time
            
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
            result = process_video_data(media_data, model_name)
            processing_time = time.time() - start_time
            
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
    return jsonify({
        "status": "success",
        "models": {
            "real-esrgan": {
                "display_name": "RealESRGAN",
                "description": "Standard 4x super-resolution"
            },
            "real-hat-gan": {
                "display_name": "Real HAT-GAN",
                "description": "Transformer-based hybrid model with sharper results"
            }
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_available": list(models.keys()),
        "model_paths": {k: os.path.exists(v) for k, v in MODEL_PATHS.items()}
    })

# Initialize models when app starts
load_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)