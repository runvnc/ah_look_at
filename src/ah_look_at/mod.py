from lib.providers.commands import command
from lib.providers.services import service
import base64
from PIL import Image
from .pdf import pdf_to_images_and_text_impl
import fitz  # PyMuPDF
import traceback
import os


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
    print("image message len: ", len(message))
    return message


ocr_prompt ="""Act as an advanced layout-aware OCR program:
   - extract the text as markdown in semantic order from the image and 
   - output ONLY the text from the image in your response
   - do NOT put your output in a markdown code block, just raw markdown throughout""" 

@service()
async def chat_ocr(model, image_path, prompt=ocr_prompt, context=None):
    """
    OCR or other text output from VLM inspecting an image.    
    """
    try:
        img_content = await examine_image(image_path, context)

        message = { "role": "user", "content": [img_content, { "text": prompt, "type": "text" } ]}
        if model is None:
            model = os.environ.get("MR_OCR_VLM", None)
        print("Starting chat stream for ocr with model: ", model)
        stream = await context.stream_chat(model, messages=[message], context=context)
        full_text = ""
        async for text_chunk in stream:
            print(text_chunk, end="")
            full_text += text_chunk
        return full_text
    except Exception as e:
        trace = traceback.format_exc()
        print(f"Error in chat_ocr: {str(e)} {trace}")
        return "Error: " + str(e) + "\n" + trace

@command()
async def prompt_vlm(image_path, prompt=ocr_prompt, context=None):
    f"""
    OCR or other text output from VLM inspecting image.
    Example:
    {{ "chat_ocr": {{ "image_path": "/absolute/path/to/file.png" }} }}

    Example (with optional prompt):
    (Note, the default prompt is shown)

    {{ "chat_ocr": {{ "image_path": "/absolute/path/to/file.png", "prompt": START_RAW
    {ocr_prompt}

    END_RAW
     }} }}
     """
    return await chat_ocr(image_path, prmopt)

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
async def examine_pdf(pdf_path, start_page=None, end_page=None, render_page_images=True, context=None):
    """For each page of a PDF in start_page - end_page (or all if not specified), 
    extracts the text and images (if possible) and, if requested, renders the page as an image. 

    In some cases, you may wish to check the number of pages with get_pdf_stats
    before calling this command.

    Also, you might investigate with render_page_images False first to find
    relevant content before running with that parameter True to get the full images
    of the relevant rendered pages.

    Example:
    
    { "examine_pdf": { "pdf_path": "/absolute/path/to/pdf.pdf",
                       "start_page": 0, "end_page": 10,
                       "render_page_images": False } }

    or

    { "examine_pdf": { "pdf_path": "/absolute/path/to/pdf.pdf" } }
    """
    try:
        w, h, pixels = await context.get_image_dimensions()
        pdf_dir = '/'.join(pdf_path.split('/')[:-1])

        if start_page is None:
            start_page = 0
        if end_page is None:
            pdf = fitz.open(pdf_path)
            end_page = pdf.page_count
        out_list = await pdf_to_images_and_text_impl(pdf_path, start_page, end_page, pdf_dir, render_page_images, max_width=w, max_height=h, max_pixels=pixels, context=context)
    except Exception as e:
        print("Error in examine_pdf: ", e)
        traceback.print_exc()
        raise e

    return out_list 


