"""
Super-resolution inference module.
"""

import os
import time
import cv2
import numpy as np
from abc import ABC, abstractmethod

class SuperResolutionModel(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    @abstractmethod
    def load_model(self):
        pass
        
    @abstractmethod
    def enhance(self, img, text_hint=None):
        pass
        
    def preprocess(self, img):
        return img
        
    def postprocess(self, output):
        return output


class RealESRGANModel(SuperResolutionModel):
    def load_model(self):
        # Import libraries here to avoid dependency issues
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
            self.model = RealESRGANer(
                scale=4,
                model_path=self.model_path,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True
            )
        except Exception as e:
            print(f"Error loading RealESRGAN model: {e}")
            self.model = None
    
    def enhance(self, img, text_hint=None):
        if self.model is None:
            # Fallback to basic upscaling
            h, w = img.shape[:2]
            return cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
            
        # Process with RealESRGAN
        try:
            output, _ = self.model.enhance(img, outscale=4)
            return output
        except Exception as e:
            print(f"Error in RealESRGAN enhancement: {e}")
            # Fallback to basic upscaling
            h, w = img.shape[:2]
            return cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)


class RealHATGANModel(SuperResolutionModel):
    def load_model(self):
        # Import libraries here to avoid dependency issues
        try:
            from basicsr.archs.hat_arch import HAT
            from realesrgan import RealESRGANer
            
            # HAT model configuration (adjust as needed for your specific model)
            model = HAT(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=16,
                depths=[6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6],
                mlp_ratio=2
            )
            
            self.model = RealESRGANer(
                scale=4,
                model_path=self.model_path,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True
            )
        except Exception as e:
            print(f"Error loading HAT model: {e}")
            self.model = None
    
    def enhance(self, img, text_hint=None):
        if self.model is None:
            # Fallback to basic upscaling
            h, w = img.shape[:2]
            return cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
            
        # Process with HAT model
        try:
            output, _ = self.model.enhance(img, outscale=4)
            return output
        except Exception as e:
            print(f"Error in HAT enhancement: {e}")
            # Fallback to basic upscaling
            h, w = img.shape[:2]
            return cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)


def create_sr_model(model_type, model_path):
    """
    Create a super-resolution model based on the specified type.
    
    Args:
        model_type: Type of model to create ('real-esrgan', 'real-hat-gan')
        model_path: Path to the model file
        
    Returns:
        Initialized super-resolution model
    """
    if model_type == 'real-esrgan':
        return RealESRGANModel(model_path)
    elif model_type == 'real-hat-gan':
        return RealHATGANModel(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def process_image(image, sr_model, text_hint=None):
    """
    Process a single image with the super-resolution model.
    
    Args:
        image: Input image as numpy array
        sr_model: Super-resolution model instance
        text_hint: Optional text hint for the model
        
    Returns:
        Enhanced image as numpy array
    """
    # Process with the model
    return sr_model.enhance(image, text_hint)


def process_video(input_path, output_path, sr_model, ocr_on_keyframes=True, keyframe_interval=30):
    """
    Process a video with the super-resolution model.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save the enhanced video
        sr_model: Super-resolution model instance
        ocr_on_keyframes: Whether to perform OCR on keyframes
        keyframe_interval: Interval between keyframes for OCR
        
    Returns:
        Dictionary with processing results
    """
    import cv2
    from ocr_utils import extract_text_with_regions
    
    start_time = time.time()
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*4, height*4))
    
    ocr_results = []
    frames_processed = 0
    
    try:
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Enhance the frame
            enhanced_frame = sr_model.enhance(frame)
            
            # Perform OCR on keyframes if requested
            if ocr_on_keyframes and frames_processed % keyframe_interval == 0:
                try:
                    ocr_result = extract_text_with_regions(enhanced_frame)
                    ocr_results.append({
                        "frame": frames_processed,
                        "timestamp": frames_processed / fps,
                        "text": ocr_result.get("text", ""),
                        "confidence": ocr_result.get("confidence", 0),
                    })
                except Exception as e:
                    print(f"Error performing OCR on frame {frames_processed}: {e}")
            
            # Write enhanced frame
            out.write(enhanced_frame)
            frames_processed += 1
            
            # Print progress every 10 frames
            if frames_processed % 10 == 0:
                print(f"Processed {frames_processed}/{frame_count} frames")
    finally:
        # Release resources
        cap.release()
        out.release()
    
    processing_time = time.time() - start_time
    
    return {
        "frames_processed": frames_processed,
        "processing_time": processing_time,
        "input_resolution": (width, height),
        "output_resolution": (width*4, height*4),
        "ocr_results": ocr_results
    }