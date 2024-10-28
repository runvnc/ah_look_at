from lib.providers.commands import command
import base64
from PIL import Image
from pdf import pdf_to_images_and_text_impl
  
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
async def examine_pdf(pdf_path, context=None):
    """For each page of a PDF, extracts the text and converts it to an image. 
    Example:
    { "examine_pdf": { "pdf_path": "/absolute/path/to/pdf.pdf" } }
    """
    w, y, pixels = await context.get_image_dimensions()
    output_path = 'output'
    out_list = await pdf_to_images_and_text_impl(pdf_path, output_path, max_width=w, max_height=h, max_pixels=pixels, context=None)
    return out_list 
