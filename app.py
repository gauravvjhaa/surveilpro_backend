import os
import time
import base64
import tempfile
import logging
import numpy as np
import cv2

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import PyTorch + Real-ESRGAN
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# (Optional) OCR import; uncomment if you want to do text extraction
# import pytesseract
# from PIL import Image

# (Optional) If you want to use MoviePy for advanced video manipulation
# import moviepy.editor as mpy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Where your .pth files are located
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Temporary folder for storing incoming/outgoing video
TEMP_DIR = tempfile.gettempdir()

# Available models
AVAILABLE_MODELS = {
    "real-esrgan": {
        "file": "RealESRGAN_x4plus.pth",
        "type": "real-esrgan",
        "display_name": "RealESRGAN",
        "description": "Standard 4× super-resolution"
    },
    "real-hat-gan": {
        "file": "Real_HAT_GAN_sharper.pth",
        "type": "real-hat-gan",
        "display_name": "Real HAT-GAN",
        "description": "Transformer-based 4× super-resolution"
    }
}

# -----------------------------------------------------------------------------
# 1) RealESRGAN Model Class
# -----------------------------------------------------------------------------
class RealESRGANModel:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=RRDBNet(
                num_in_ch=3, 
                num_out_ch=3, 
                num_feat=64, 
                num_block=23, 
                gc=32
            ),
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=self.device
        )
    
    def enhance(self, img_bgr):
        # Convert from BGR (OpenCV) to RGB (PyTorch)
        img_rgb = img_bgr[:, :, ::-1]
        with torch.no_grad():
            output_rgb, _ = self.upsampler.enhance(img_rgb, outscale=4)
        # Convert back to BGR
        return output_rgb[:, :, ::-1]

# -----------------------------------------------------------------------------
# 2) Real-HAT-GAN Model Class (Placeholder)
#    Replace with actual code if you have the correct .pth + architecture
# -----------------------------------------------------------------------------
class RealHATGANModel:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        logger.warning(
            "RealHATGANModel is a placeholder. If you have the real architecture, "
            "load it similarly to RealESRGANModel."
        )
        # e.g.:
        # self.model = SomeTransformer(...)
        # checkpoint = torch.load(model_path, map_location=self.device)
        # self.model.load_state_dict(checkpoint)
        # self.model.to(self.device).eval()

    def enhance(self, img_bgr):
        # For demonstration, just do a 4× OpenCV resize
        h, w = img_bgr.shape[:2]
        return cv2.resize(img_bgr, (w*4, h*4), interpolation=cv2.INTER_CUBIC)

# -----------------------------------------------------------------------------
# 3) Load Model Helper
# -----------------------------------------------------------------------------
def load_sr_model(model_key, device='cuda'):
    if model_key not in AVAILABLE_MODELS:
        logger.warning(f"Unknown model '{model_key}', defaulting to 'real-esrgan'.")
        model_key = "real-esrgan"
    info = AVAILABLE_MODELS[model_key]
    model_path = os.path.join(MODEL_DIR, info["file"])

    if info["type"] == "real-esrgan":
        return RealESRGANModel(model_path, device=device)
    elif info["type"] == "real-hat-gan":
        return RealHATGANModel(model_path, device=device)
    else:
        logger.warning(f"Unknown model type {info['type']}, using RealESRGAN fallback.")
        return RealESRGANModel(model_path, device=device)

# -----------------------------------------------------------------------------
# (Optional) OCR Helper - If you want to extract text
# -----------------------------------------------------------------------------
# def extract_text_from_bgr(img_bgr):
#     # Convert BGR -> RGB
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     pil_img = Image.fromarray(img_rgb)
#     text = pytesseract.image_to_string(pil_img)
#     return text

# -----------------------------------------------------------------------------
# 4) Image Processing
# -----------------------------------------------------------------------------
def process_image_data(base64_image, model_key="real-esrgan"):
    """
    1) Decode base64 -> BGR
    2) Load chosen SR model
    3) Upscale (4×)
    4) Return base64 of the enhanced PNG
    """
    try:
        image_bytes = base64.b64decode(base64_image)
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode returned None (invalid image data).")
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        return {"status": "error", "message": f"Image decode error: {str(e)}"}

    # (Optional) original OCR
    # orig_text = extract_text_from_bgr(image)

    sr_model = load_sr_model(model_key, device='cuda')
    try:
        enhanced = sr_model.enhance(image)
    except Exception as e:
        logger.error(f"SR error: {e}")
        return {"status": "error", "message": f"SR error: {str(e)}"}
    
    # (Optional) OCR on enhanced
    # enhanced_text = extract_text_from_bgr(enhanced)

    success, buffer = cv2.imencode('.png', enhanced)
    if not success:
        return {"status": "error", "message": "Could not encode enhanced image."}
    enhanced_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "status": "success",
        "enhanced_image": enhanced_b64,
        "processing_info": {
            "orig_shape": [image.shape[0], image.shape[1]],
            "enhanced_shape": [enhanced.shape[0], enhanced.shape[1]],
            "model_used": model_key
        },
        # "ocr_info": {
        #    "original_text": orig_text,
        #    "enhanced_text": enhanced_text
        # }
    }

# -----------------------------------------------------------------------------
# 5) Video Processing
# -----------------------------------------------------------------------------
def process_video_data(base64_video, model_key="real-esrgan"):
    """
    1) Decode base64 -> .mp4 in a temp file
    2) Read frames via OpenCV
    3) Upscale each frame 4×
    4) Write to output .mp4
    5) Return base64 of the new mp4
    """
    # Save paths
    ts = int(time.time())
    input_path = os.path.join(TEMP_DIR, f"input_{ts}.mp4")
    output_path = os.path.join(TEMP_DIR, f"output_{ts}.mp4")

    try:
        video_bytes = base64.b64decode(base64_video)
    except Exception as e:
        logger.error(f"Base64 decode error: {e}")
        return {"status": "error", "message": f"Video decode error: {str(e)}"}

    try:
        # Write the input video
        with open(input_path, 'wb') as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open input video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w*4, h*4))

        sr_model = load_sr_model(model_key, device='cuda')

        frames_processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            enhanced_frame = sr_model.enhance(frame)
            out.write(enhanced_frame)
            frames_processed += 1

        cap.release()
        out.release()

        # Read result as base64
        with open(output_path, 'rb') as f:
            out_bytes = f.read()
        enhanced_video_b64 = base64.b64encode(out_bytes).decode('utf-8')

        return {
            "status": "success",
            "enhanced_video": enhanced_video_b64,
            "processing_info": {
                "frames_processed": frames_processed,
                "input_resolution": [h, w],
                "output_resolution": [h*4, w*4],
                "model_used": model_key
            }
        }
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        return {"status": "error", "message": f"Video processing error: {str(e)}"}
    finally:
        # Cleanup
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

# -----------------------------------------------------------------------------
# 6) Flask Routes
# -----------------------------------------------------------------------------
@app.route('/process_media', methods=['POST'])
def process_media():
    """
    Receives JSON:
    {
      "media_data": "<base64>",
      "media_type": "image" or "video",
      "model": "real-esrgan" or "real-hat-gan"
    }
    Returns JSON with either "enhanced_image" or "enhanced_video" in base64.
    """
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON."}), 400

    data = request.get_json()
    media_data = data.get('media_data')
    media_type = data.get('media_type')
    model_key = data.get('model', 'real-esrgan')

    if not media_data or not media_type:
        return jsonify({
            "status": "error",
            "message": "Missing 'media_data' or 'media_type'."
        }), 400

    start_time = time.time()

    # Decide image or video
    if media_type.lower() == 'image':
        result = process_image_data(media_data, model_key)
    elif media_type.lower() == 'video':
        result = process_video_data(media_data, model_key)
    else:
        return jsonify({
            "status": "error", 
            "message": f"Unsupported media_type '{media_type}'."
        }), 400

    if result.get('status') == 'error':
        # Some internal error
        return jsonify(result), 500

    elapsed = time.time() - start_time
    result['processing_time'] = elapsed
    return jsonify(result)

@app.route('/models', methods=['GET'])
def list_models():
    """List available models."""
    return jsonify({"status": "success", "models": AVAILABLE_MODELS})

@app.route('/health', methods=['GET'])
def health():
    """Simple health check."""
    return jsonify({"status": "healthy"})

@app.route('/test_connection', methods=['GET', 'POST'])
def test_connection():
    """Simple test endpoint for GET/POST."""
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            return jsonify({
                "status": "success",
                "message": "POST with JSON received.",
                "received_data": data
            })
        else:
            return jsonify({"status": "success", "message": "POST but no JSON."})
    else:
        return jsonify({"status": "success", "message": "GET: connected."})

# -----------------------------------------------------------------------------
# 7) Entry Point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Render sets PORT in the environment
    port = int(os.environ.get('PORT', 5000))
    # Flask dev server (fine for small demos); for production, use Gunicorn:
    #   gunicorn app:app --bind 0.0.0.0:$PORT
    app.run(host='0.0.0.0', port=port, debug=False)
