import fitz  # PyMuPDF
import os
import cv2
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

async def pdf_to_images_and_text_impl(pdf_path, output_dir, max_width=1568, max_height=1568, max_pixels=1192464, context=None, debug=False):
    """Generate a constrained overview image for each page in the PDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files
        max_width: Maximum width constraint for output images
        max_height: Maximum height constraint for output images
        max_pixels: Maximum total pixels constraint
        context: Context object for image formatting
        debug: If True, writes debug information to files (default: False)
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_data = []
    # Determine optimal DPI based on constraints
    dpi = 72  # Start with default DPI
    zoom = 1  # Initial zoom factor for 72 DPI

    try:
        for page_num in range(doc.page_count):
            start_time = time.time()
            page = doc[page_num]
            
            # Extract text from the page
            text = page.get_text()
            
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
            while True:
                print('------- zoom iteration -------')
                print('DPI: ', dpi)
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=content_rect)
                if (pix.width >= max_width or pix.height >= max_height or 
                    pix.width * pix.height >= max_pixels) or zoom >= 5:
                    dpi -= 50
                    zoom = dpi / 72
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=content_rect)
                    break
                dpi += 50  # Increment DPI
                zoom = dpi / 72  # iUpdate zoom factor accordingly
            print("Elapsed time for zoom: ", (time.time() - st_zoom) * 1000)

            st_encode = time.time()
            try:
                # Convert final pixmap to image
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
                # Instead of writing to disk and reading back, use BytesIO
                img_byte_array = BytesIO()
                # Convert to BGR for cv2.imwrite
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                # Encode image to png format
                _, encoded_img = cv2.imencode('.png', img_bgr)
                img_byte_array.write(encoded_img.tobytes())
                img_byte_array.seek(0)

                # Still save the file for reference
                #overview_filename = f"{output_dir}/page_{page_num + 1}_overview.png"
                #with open(overview_filename, 'wb') as f:
                #    f.write(img_byte_array.getvalue())

                page_info = {
                    "page_num": page_num + 1,
                    "text": text,
                    #"image_filename": overview_filename,
                    "image_data": "in next message",
                    "dimensions": (img_array.shape[1], img_array.shape[0]),
                    "dpi_used": dpi
                }
                page_data.append(page_info)
                print("Encode tiempo: ", (time.time() - st_encode) * 1000)

                # Create PIL Image from BytesIO instead of file
                img_byte_array.seek(0)
                pil_image = Image.open(img_byte_array)
                formatted_image_message = await context.format_image_message(pil_image)
                page_data.append(formatted_image_message)
            
            finally:
                # Clean up resources
                pix = None
                img_array = None
                img_bgr = None
                encoded_img = None
                img_byte_array.close()

            print("Elapsed time: ", (time.time() - start_time) * 1000)

 
    finally:
        # Ensure document is closed
        doc.close()
    
    if debug:
        await write_debug_file(page_data)
    return page_data


def generate_zoomed_crop(pdf_path, output_dir, page_num, upper_left, lower_right, dpi=300):
    """Generate a high-DPI zoomed crop of a specified area on a given page."""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    
    try:
        # Select the page and define initial crop area
        page = doc[page_num - 1]  # Zero-based index
        initial_crop_rect = fitz.Rect(upper_left, lower_right)
        
        # Get initial low-res pixmap for smart crop calculation
        initial_zoom = 1  # 72 DPI
        initial_pix = page.get_pixmap(matrix=fitz.Matrix(initial_zoom, initial_zoom), clip=initial_crop_rect)
        
        try:
            # Get content bounds from low-res version
            img_array = np.frombuffer(initial_pix.samples, dtype=np.uint8).reshape(initial_pix.height, initial_pix.width, initial_pix.n)
            if initial_pix.n == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Convert to grayscale and find content bounds
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            coords = cv2.findNonZero(thresh)
            
            if coords is None:
                # No content found, use full crop area
                final_crop_rect = initial_crop_rect
            else:
                # Get exact bounds without margins
                x, y, w, h = cv2.boundingRect(coords)
                
                # Convert to PDF coordinates within the initial crop area
                x_ratio = initial_crop_rect.width / initial_pix.width
                y_ratio = initial_crop_rect.height / initial_pix.height
                
                # Create content rect with exact bounds
                final_crop_rect = fitz.Rect(
                    initial_crop_rect.x0 + x * x_ratio,
                    initial_crop_rect.y0 + y * y_ratio,
                    initial_crop_rect.x0 + (x + w) * x_ratio,
                    initial_crop_rect.y0 + (y + h) * y_ratio
                )
        finally:
            # Clean up resources
            initial_pix = None
            img_array = None
            gray = None
            thresh = None
        
        # Apply high-DPI scaling for detailed rendering
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix, clip=final_crop_rect)
        
        try:
            # Convert final pixmap to image
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Save the zoomed crop
            zoomed_filename = f"{output_dir}/page_{page_num}_zoomed_crop.png"
            cv2.imwrite(zoomed_filename, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            
            return zoomed_filename
        finally:
            # Clean up resources
            pix = None
            img_array = None
            
    finally:
        # Ensure document is closed
        doc.close()
