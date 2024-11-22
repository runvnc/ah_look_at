import fitz  # PyMuPDF
import math
import cv2
import os
import numpy as np
from io import BytesIO
import base64
from PIL import Image
import json
import time

async def write_debug_file(data):
    page_num = 1
    for page in data:
        if 'type' in page:
            with open(f"output/page_{page_num}_image_msg.json", "w") as f:
                f.write(json.dumps(page))
                print("Wrote image message to file: ", f.name)
        page_num += 1

def extract_page_images(doc, page, output_dir, page_num):
    """Extract all images from a PDF page and save to files.
    
    Args:
        doc: fitz.Document - The PDF document
        page: fitz.Page - The page to extract images from
        output_dir: str - Directory to save images
        page_num: int - Current page number (0-based)
        
    Returns:
        list: List of dictionaries containing image info
    """
    #images_dir = os.path.join(output_dir, "images")
    images_dir = output_dir
    os.makedirs(images_dir, exist_ok=True)
    
    extracted_images = []
    
    # Get list of images on the page
    image_list = page.get_images()
    
    for img_index, img_info in enumerate(image_list):
        try:
            xref = img_info[0]  # Get image reference number
            base_image = doc.extract_image(xref)
            
            if base_image:
                image_ext = base_image["ext"]  # Image extension (jpeg, png, etc)
                image_bytes = base_image["image"]  # Image data
                
                #filename = f"page_{page_num+1}_img_{img_index+1}_{base_image['width']}x{base_image['height']}.{image_ext}"

                img_name = img_info[7] if len(img_info) > 7 else 'unnamed'
                colorspace = img_info[5]
                #colorspace = base_image.get('colorspace', 'unknown')
                filename = f"page_{page_num+1}_img_{img_index+1}_{base_image['width']}x{base_image['height']}_{colorspace}_{img_name}.{image_ext}"
                filename = ''.join(c for c in filename if c.isalnum() or c in '._-')

                filepath = os.path.abspath(os.path.join(images_dir, filename))
                
                # Save image to file
                with open(filepath, 'wb') as img_file:
                    img_file.write(image_bytes)
                
                # Record image info
                extracted_images.append({
                    "filepath": filepath,
                    "size": len(image_bytes),
                    "type": image_ext,
                    "colorspace": colorspace,
                    "width": base_image.get("width"),
                    "height": base_image.get("height")
                })
                
        except Exception as e:
            print(f"Error extracting image {img_index} from page {page_num+1}: {e}")
            continue
            
    return extracted_images


async def pdf_to_images_and_text_impl(pdf_path, start_page, end_page, output_dir, render_page_images, max_width=1568, max_height=1568, max_pixels=1192464, context=None, debug=False):
    """Generate a constrained overview image for each page in the PDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files
        start_page: First page to process (0-based)
        end_page: Last page to process (0-based)
        max_width: Maximum width constraint for output images
        max_height: Maximum height constraint for output images
        max_pixels: Maximum total pixels constraint
        context: Context object for image formatting
        debug: If True, writes debug information to files (default: False)
    """
    os.makedirs(output_dir, exist_ok=True)
    page_data = []
    # Cache for zoom calculations
    last_content_dims = None
    last_zoom = None
    last_dpi = None
    # Reusable BytesIO buffer
    reusable_buffer = BytesIO()
    doc = fitz.open(pdf_path)
    
    try:
        if start_page == end_endpage:
            end_page += 1
        for page_num in range(start_page, end_page):
            start_time = time.time()
            page = doc[page_num]
            
            text = page.get_text()
            
            extracted_images = extract_page_images(doc, page, output_dir, page_num)
            img_array = None
            dimensions = None
            dpi = None

            if render_page_images:

                initial_zoom = 1  # 72 DPI
                initial_pix = page.get_pixmap(matrix=fitz.Matrix(initial_zoom, initial_zoom))
            
                st_bounds = time.time()
                try:
                    # Get content bounds from low-res version
                    img_array = np.frombuffer(initial_pix.samples, dtype=np.uint8).reshape(initial_pix.height, initial_pix.width, initial_pix.n)
                    # Convert to RGB only if needed
                    if initial_pix.n == 4:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                    
                    # Convert to grayscale and find content bounds
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
                    coords = cv2.findNonZero(thresh)
                    
                    if coords is None:
                        # No content found, use full page
                        content_rect = page.rect
                    else:
                        # Get exact bounds without margins
                        x, y, w, h = cv2.boundingRect(coords)
                        
                        # Convert to PDF coordinates
                        x_ratio = page.rect.width / initial_pix.width
                        y_ratio = page.rect.height / initial_pix.height
                        
                        # Create content rect with exact bounds
                        content_rect = fitz.Rect(
                            page.rect.x0 + x * x_ratio,
                            page.rect.y0 + y * y_ratio,
                            page.rect.x0 + (x + w) * x_ratio,
                            page.rect.y0 + (y + h) * y_ratio
                        )
                finally:
                    # Clean up initial pixmap
                    initial_pix = None
                    img_array = None
                    gray = None
                    thresh = None

                print("Elapsed time for bounds: ", (time.time() - st_bounds) * 1000)
                st_zoom = time.time()
                
                # Get current content dimensions
                content_width = content_rect.width
                content_height = content_rect.height
                current_dims = (content_width, content_height)
                
                # Check if dimensions match the previous page (within 1% tolerance)
                if last_content_dims and last_zoom and last_dpi:
                    width_diff = abs(current_dims[0] - last_content_dims[0]) / last_content_dims[0]
                    height_diff = abs(current_dims[1] - last_content_dims[1]) / last_content_dims[1]
                    
                    if width_diff < 0.01 and height_diff < 0.01:
                        # Reuse previous zoom values
                        zoom = last_zoom
                        dpi = last_dpi
                        print("Reusing zoom values from previous page")
                    else:
                        # Calculate new zoom values
                        width_zoom = max_width / content_width
                        height_zoom = max_height / content_height
                        
                        # Initial zoom estimation (take the minimum to satisfy both constraints)
                        zoom = min(width_zoom, height_zoom)
                        
                        # Factor in max_pixels constraint
                        pixels_zoom = math.sqrt(max_pixels / (content_width * content_height))
                        zoom = min(zoom, pixels_zoom)
                        
                        # Convert to DPI and ensure it's within reasonable bounds
                        dpi = min(max(72, zoom * 72), 360)  # Cap at 360 DPI
                        zoom = dpi / 72
                else:
                    # First page, calculate zoom values
                    width_zoom = max_width / content_width
                    height_zoom = max_height / content_height
                    zoom = min(width_zoom, height_zoom)
                    pixels_zoom = math.sqrt(max_pixels / (content_width * content_height))
                    zoom = min(zoom, pixels_zoom)
                    dpi = min(max(72, zoom * 72), 360)
                    zoom = dpi / 72
                
                # Create final pixmap with calculated zoom
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=content_rect)
                
                # Verify constraints and adjust if needed
                if pix.width * pix.height > max_pixels or zoom >= 5:
                    zoom = zoom * 0.8  # Reduce by 20% if still too large
                    dpi = zoom * 72
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=content_rect)
                
                print("DPI: ", dpi)
                # Cache the values for next page
                last_content_dims = current_dims
                last_zoom = zoom
                last_dpi = dpi
                
                print("Elapsed time for zoom: ", (time.time() - st_zoom) * 1000)
                st_encode = time.time()
                try:
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    
                    if pix.n == 4:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                    
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    _, encoded_img = cv2.imencode('.png', img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    
                    reusable_buffer.seek(0)
                    reusable_buffer.truncate(0)
                    reusable_buffer.write(encoded_img.tobytes())
                    reusable_buffer.seek(0)
                
                    pil_image = Image.open(reusable_buffer)
                
                    with open(f"output/page_{page_num}_image.png", "wb") as f:
                        f.write(encoded_img)
                        print("Wrote image to file: ", f.name)

                finally:
                    pix = None
                    img_bgr = None
                        
            page_info = {
                "page_num": page_num + 1,
                "text": text,
                "dpi_used": dpi,
                "extracted_images": extracted_images
            }
            if render_page_images:
                page_info["dimensions"] = (img_array.shape[1], img_array.shape[0])
                page_info["image_data"] = 'in next message'

            page_data.append(page_info)
            
            if render_page_images:
                formatted_image_message = await context.format_image_message(pil_image)
                page_data.append(formatted_image_message)
                
                print("Encode time: ", (time.time() - st_encode) * 1000)
            
            img_array = None
            pil_image = None
            print("Elapsed time: ", (time.time() - start_time) * 1000)

    finally:
        # Ensure document is closed and buffer cleaned up
        doc.close()
        reusable_buffer.close()
    
    if debug:
        await write_debug_file(page_data)
    return page_data
