# SurveilPro - AI-Enhanced Surveillance Media Processing

A secure, Flask-based backend for enhancing surveillance footage using super-resolution models, with built-in OCR capabilities.

## 🌟 Features

- **End-to-End Encryption**: All media transmitted with RSA encryption
- **Super-Resolution Enhancement**: Upgrade low-resolution images and videos
- **OCR Processing**: Extract text from surveillance footage
- **Dual Processing Pipeline**: Handle both images and videos
- **Modular Architecture**: Easy to swap models and components

## 📋 Requirements

- Python 3.8+ 
- PyTorch 1.9+
- OpenCV 4.5+
- Tesseract OCR
- See `requirements.txt` for full dependencies

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/surveilpro.git
cd surveilpro
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Install Tesseract OCR:
   - **Ubuntu**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

4. Download the model files:
```bash
mkdir -p model
# Download Real-ESRGAN model
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O model/RealESRGAN_x4plus.pth
```

## 🚀 Usage

1. Start the server:
```bash
python app.py
```

2. The server will automatically generate RSA keys in the `keys` directory if they don't exist.

3. Send requests to the `/process_media` endpoint:
```
POST /process_media
Content-Type: application/json

{
    "encrypted_media": "<base64 RSA encrypted blob>",
    "media_type": "image"  // or "video"
}
```

4. The response will be in the format:
```json
{
    "status": "success",
    "encrypted_result": "<base64 RSA encrypted blob>",
    "processing_time": 1.23
}
```

5. Check server health:
```
GET /health
```

## 🔐 Security Notes

- The system uses RSA encryption for secure media transfer
- RSA keys are generated on first run
- In a production environment, key management should be handled more securely
- All temporary files are deleted after processing

## 🖥️ API Endpoints

### /process_media

**Method**: POST

**Request**:
```json
{
    "encrypted_media": "<base64 RSA encrypted blob>",
    "media_type": "image"  // or "video"
}
```

**Response**:
```json
{
    "status": "success",
    "encrypted_result": "<base64 RSA encrypted blob>",
    "processing_time": 1.23
}
```

The decrypted `encrypted_result` contains:

For images:
```json
{
    "enhanced_image": "<base64 image data>",
    "original_ocr": {
        "text": "extracted text before enhancement",
        "confidence": 85.2
    },
    "enhanced_ocr": {
        "text": "extracted text after enhancement",
        "confidence": 94.7
    },
    "processing_info": {
        "original_size": [480, 640, 3],
        "enhanced_size": [1920, 2560, 3],
        "enhancement_factor": 4.0
    }
}
```

For videos:
```json
{
    "enhanced_video": "<base64 video data>",
    "ocr_results": [
        {
            "frame": 0,
            "timestamp": 0.0,
            "text": "extracted text",
            "confidence": 87.3
        },
        // More frames...
    ],
    "processing_info": {
        "frames_processed": 120,
        "processing_time": 45.2,
        "input_resolution": [640, 480],
        "output_resolution": [2560, 1920]
    }
}
```

### /health

**Method**: GET

**Response**:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "encryption_ready": true
}
```

## 🧩 Architecture

The backend is structured into four main components:

1. **Flask API** (`app.py`): Handles HTTP requests and manages the processing pipeline
2. **Encryption** (`encryption_utils.py`): Manages RSA encryption/decryption
3. **OCR Processing** (`ocr_utils.py`): Text extraction and confidence scoring
4. **Super-Resolution** (`sr_inference.py`): Image/video enhancement

## 🔄 Processing Pipeline

### Image Processing:
1. Decrypt image data
2. Run initial OCR
3. Pass image and text hints to super-resolution model
4. Enhance the image
5. Run OCR again on enhanced image
6. Package and encrypt results

### Video Processing:
1. Decrypt video data
2. Extract frames
3. Process each frame with super-resolution
4. Optional OCR on keyframes
5. Reconstruct enhanced video
6. Package and encrypt results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.