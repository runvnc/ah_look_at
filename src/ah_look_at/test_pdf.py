from pdf import pdf_to_images_and_text_impl, generate_zoomed_crop
import asyncio
from io import BytesIO
import base64
from PIL import Image
import json

# Usage example
pdf_path = '/files/home/runvnc/3pages.pdf'  # Replace with your PDF file path
output_dir = 'output'      # Replace with your desired output directory

class TestVisionService:
    def __init__(self):
        pass

    async def format_image_message(self, pil_image, context=None):
        print("saving")
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
 
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": f"image/png",
                "data": image_base64
            }
        }

    async def get_image_dimensions(self, context=None):
        return 1568, 1568, 1192464 

service = TestVisionService()

async def run_tests():

    overview_data = await pdf_to_images_and_text_impl(pdf_path, output_dir, context=service)
    page_num = 1
    for page in overview_data:
        if 'type' in page:
            with open(f"output/page_{page_num}_image_msg.json", "w") as f:
                f.write(json.dumps(page))
                print("Wrote image message to file: ", f.name)
        else:
            print(f"Page {page_num} - Image file: {page['image_filename']}, Dimensions: {page['dimensions']}, DPI: {page['dpi_used']}")
            print(page['text'])
            page_num = page['page_num']

    # Generate a zoomed crop for a specific area on page 1
    #upper_left = (100, 100)  # X, Y coordinates of the upper-left corner
    #lower_right = (300, 300)  # X, Y coordinates of the lower-right corner
    #zoomed_crop_path = generate_zoomed_crop(pdf_path, output_dir, page_num=1, upper_left=upper_left, lower_right=lower_right)
    #print(f"Zoomed crop saved to {zoomed_crop_path}")


# Run the tests
asyncio.run(run_tests())

