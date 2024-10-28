from pdf import generate_constrained_overview_image, generate_zoomed_crop

# Usage example
pdf_path = '/files/home/runvnc/3pages.pdf'  # Replace with your PDF file path
output_dir = 'output'      # Replace with your desired output directory

# Generate constrained overview images
overview_data = generate_constrained_overview_image(pdf_path, output_dir)
for page in overview_data:
    print(f"Page {page['page_num']} - Overview: {page['overview_image']}, Dimensions: {page['overview_dimensions']}, DPI: {page['dpi_used']}")

# Generate a zoomed crop for a specific area on page 1
upper_left = (100, 100)  # X, Y coordinates of the upper-left corner
lower_right = (300, 300)  # X, Y coordinates of the lower-right corner
zoomed_crop_path = generate_zoomed_crop(pdf_path, output_dir, page_num=1, upper_left=upper_left, lower_right=lower_right)
print(f"Zoomed crop saved to {zoomed_crop_path}")

