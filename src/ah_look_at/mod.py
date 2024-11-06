from lib.providers.commands import command
import base64
from PIL import Image
from .pdf import pdf_to_images_and_text_impl
import fitz  # PyMuPDF

@command()
async def examine_image(full_image_path, context=None):
    """
    This will read an image and insert it into the chat context.

    Example:

    { "examine_image": { "full_image_path": "/path/to/image.jpg" } }

    """
    print("loading image")
    image = Image.open(full_image_path)
    print(image)
    message = await context.format_image_message(image)
    print("image message: ", message)
    return message


@command()
async def get_pdf_stats(pdf_path, context=None):
    """
    Get the number of pages and metadata of a PDF file.
    """
    pdf = fitz.open(pdf_path)
    num_pages = pdf.page_count
    metadata = pdf.metadata
    return {"num_pages": num_pages, "metadata": metadata}


@command()
async def examine_pdf(pdf_path, start_page, end_page, render_page_images=True, context=None):
    """For each page of a PDF in start_page - end_page, extracts the text and images (if possible) and,
    if requested, renders the page as an image. 
    Example:
    { "examine_pdf": { "pdf_path": "/absolute/path/to/pdf.pdf" } }
    """
    w, h, pixels = await context.get_image_dimensions()
    output_path = 'output'
    out_list = await pdf_to_images_and_text_impl(pdf_path, start_page, end_page, output_path, render_page_images, max_width=w, max_height=h, max_pixels=pixels, context=context)
    return out_list 


