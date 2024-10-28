import fitz  # PyMuPDF
import os
import cv2
import numpy as np

def smart_crop_image(pix):
    """Apply smart cropping to remove white margins while preserving all content."""
    # Convert PyMuPDF pixmap to OpenCV format
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Threshold to separate content from background
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Find non-zero points (content)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img_array  # Return original if no content found
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add small margin (1% of each dimension)
    margin_x = int(w * 0.01)
    margin_y = int(h * 0.01)
    
    # Ensure margins don't exceed image bounds
    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    w = min(img_array.shape[1] - x, w + 2 * margin_x)
    h = min(img_array.shape[0] - y, h + 2 * margin_y)
    
    # Crop and return
    return img_array[y:y+h, x:x+w]


