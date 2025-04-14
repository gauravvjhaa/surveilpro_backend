"""
Optical Character Recognition utilities using Tesseract and OpenCV.
"""

import cv2
import numpy as np
import pytesseract
from typing import Dict, Any, List, Tuple


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for better OCR results.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Noise removal
    denoise = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Thresholding to binarize
    _, binary = cv2.threshold(denoise, 0, 255, 
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Additional morphological operations for text enhancement
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return opening


def extract_text_from_image(
    image: np.ndarray, 
    preprocess: bool = True
) -> Dict[str, Any]:
    """
    Extract text from an image using Tesseract OCR.
    
    Args:
        image: Input image as numpy array
        preprocess: Whether to preprocess the image first
        
    Returns:
        Dictionary with OCR results:
        {
            'text': extracted text,
            'confidence': average confidence score,
            'details': detailed OCR data
        }
    """
    # Preprocess the image if requested
    if preprocess:
        processed_image = preprocess_for_ocr(image)
    else:
        processed_image = image
    
    # Use Tesseract OCR to get text and confidence data
    # Get detailed output with confidence scores
    ocr_data = pytesseract.image_to_data(
        processed_image, 
        output_type=pytesseract.Output.DICT
    )
    
    # Extract confidence scores for valid text entries
    confidences = [
        int(conf) for conf, text in zip(ocr_data['conf'], ocr_data['text'])
        if text.strip() and float(conf) > 0  # Filter out empty strings and -1 confidences
    ]
    
    # Calculate average confidence
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Combine all text blocks
    full_text = ' '.join([text for text in ocr_data['text'] if text.strip()])
    
    return {
        'text': full_text,
        'confidence': avg_confidence,
        'details': ocr_data
    }


def detect_regions_of_interest(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect potential regions containing text or license plates.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        List of (x, y, width, height) bounding boxes
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and aspect ratio (likely text areas)
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = w * h
        
        # Filter based on size and aspect ratio
        # License plates typically have aspect ratio between 2.0 and 5.0
        # Text regions are typically wider than tall
        if area > 500 and (0.8 < aspect_ratio < 6.0):
            regions.append((x, y, w, h))
    
    return regions


def extract_text_with_regions(image: np.ndarray) -> Dict[str, Any]:
    """
    Extract text by first detecting regions of interest, then applying OCR.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary with OCR results and region data
    """
    # Get potential text regions
    regions = detect_regions_of_interest(image)
    
    all_text = []
    all_confidences = []
    region_results = []
    
    # Process each region
    for i, (x, y, w, h) in enumerate(regions):
        # Extract region with a small margin
        margin = 5
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(image.shape[1], x + w + margin)
        y_end = min(image.shape[0], y + h + margin)
        
        region_image = image[y_start:y_end, x_start:x_end]
        
        # Skip tiny regions
        if region_image.size == 0 or region_image.shape[0] < 10 or region_image.shape[1] < 10:
            continue
            
        # Process this region
        ocr_result = extract_text_from_image(region_image)
        
        if ocr_result['text']:  # Only consider non-empty results
            all_text.append(ocr_result['text'])
            all_confidences.append(ocr_result['confidence'])
            
            region_results.append({
                'region': (x, y, w, h),
                'text': ocr_result['text'],
                'confidence': ocr_result['confidence']
            })
    
    # If no text found in regions, try whole image
    if not all_text:
        whole_image_ocr = extract_text_from_image(image)
        return whole_image_ocr
    
    # Calculate overall result
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    combined_text = ' '.join(all_text)
    
    return {
        'text': combined_text,
        'confidence': avg_confidence,
        'regions': region_results
    }


def detect_license_plates(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Specialized function to detect and OCR license plates in images.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        List of detected license plates with text and positions
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    license_plates = []
    
    # Filter for rectangular contours with appropriate aspect ratio
    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # If we have a quadrilateral
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # License plates typically have aspect ratio between 2 and 5
            if 1.5 < aspect_ratio < 5.0 and w > 60 and h > 20:
                # Extract the region
                plate_img = image[y:y+h, x:x+w]
                
                # OCR the plate
                ocr_result = extract_text_from_image(plate_img, preprocess=True)
                
                if ocr_result['text']:
                    license_plates.append({
                        'text': ocr_result['text'],
                        'confidence': ocr_result['confidence'],
                        'box': (x, y, w, h)
                    })
    
    return license_plates


def process_keyframes_for_ocr(
    video_path: str, 
    interval: int = 30
) -> List[Dict[str, Any]]:
    """
    Extract and OCR keyframes from a video.
    
    Args:
        video_path: Path to the video file
        interval: Process every Nth frame
        
    Returns:
        List of OCR results for keyframes
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    results = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every Nth frame
        if frame_count % interval == 0:
            ocr_result = extract_text_with_regions(frame)
            
            results.append({
                'frame': frame_count,
                'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS),
                'ocr_result': ocr_result
            })
            
        frame_count += 1
    
    cap.release()
    return results
