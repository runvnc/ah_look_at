from lib.providers.commands import command
import base64
from PIL import Image

@command()
async def examine_image(full_image_path, context=None):
    """
    This will read an image and insert it into the chat context.

    Example:

    { "examine_image": { "full_image_path": "/path/to/image.jpg" } }

    """
    image = Image.open(full_image_path)
    message = context.format_image_message(image)
    return message
