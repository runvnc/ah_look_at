import fitz  # PyMuPDF
import os
import cv2
import numpy as np

def pdf_to_images_and_text_impl(pdf_path, output_dir, max_width=1568, max_height=1568, max_pixels=1192464):
    """Generate a constrained overview image for each page in the PDF."""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_data = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        
        # Extract text from the page
        text = page.get_text()
        
        initial_zoom = 1  # 72 DPI
        initial_pix = page.get_pixmap(matrix=fitz.Matrix(initial_zoom, initial_zoom))
        
        # Get content bounds from low-res version
        img_array = np.frombuffer(initial_pix.samples, dtype=np.uint8).reshape(initial_pix.height, initial_pix.width, initial_pix.n)
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

        # Determine optimal DPI based on constraints
        dpi = 72  # Start with default DPI
        zoom = 1  # Initial zoom factor for 72 DPI

        while True:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=content_rect)
            if (pix.width >= max_width or pix.height >= max_height or 
                pix.width * pix.height >= max_pixels) or zoom >= 5:
                dpi -= 50
                zoom = dpi / 72
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=content_rect)
                break
            dpi += 50  # Increment DPI
            zoom = dpi / 72  # Update zoom factor accordingly
        
        # Convert final pixmap to image
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Save the constrained overview image
        overview_filename = f"{output_dir}/page_{page_num + 1}_overview.png"
        cv2.imwrite(overview_filename, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        # Collect data for the overview image
        page_info = {
            "page_num": page_num + 1,
            "text": text,
            "overview_image": overview_filename,
            "overview_dimensions": (img_array.shape[1], img_array.shape[0]),
            "dpi_used": dpi
        }
        page_data.append(page_info)

    doc.close()
    return page_data


def generate_zoomed_crop(pdf_path, output_dir, page_num, upper_left, lower_right, dpi=300):
    """Generate a high-DPI zoomed crop of a specified area on a given page."""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    
    # Select the page and define initial crop area
    page = doc[page_num - 1]  # Zero-based index
    initial_crop_rect = fitz.Rect(upper_left, lower_right)
    
    # Get initial low-res pixmap for smart crop calculation
    initial_zoom = 1  # 72 DPI
    initial_pix = page.get_pixmap(matrix=fitz.Matrix(initial_zoom, initial_zoom), clip=initial_crop_rect)
    
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
    
    # Apply high-DPI scaling for detailed rendering
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix, clip=final_crop_rect)
    
    # Convert final pixmap to image
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Save the zoomed crop
    zoomed_filename = f"{output_dir}/page_{page_num}_zoomed_crop.png"
    cv2.imwrite(zoomed_filename, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    doc.close()
    return zoomed_filename
    
