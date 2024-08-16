import fitz  # PyMuPDF
import base64
import json
import os
from pathlib import Path
import multiprocessing

# def process_pdf_to_png(job, output_file):
#     pdf_base64 = job['pdf']
#     # job_id = job['job_id']

#     # Decode the base64 string to binary PDF data
#     pdf_data = base64.b64decode(pdf_base64)

#     # Load the PDF from binary data
#     pdf_document = fitz.open("pdf", pdf_data)

#     # Render the first page into a PNG image at 300 DPI
#     page = pdf_document.load_page(0)  # Assuming the PDF only has one page
    
#     pix = page.get_pixmap(dpi=300) # highest resolution
    
#     png_data = pix.tobytes("png")

#     # Encode the PNG data to base64
#     png_base64 = base64.b64encode(png_data).decode('utf-8')

#     # Update the job with the base64 encoded PNG
#     job['image'] = png_base64

#     # Write the updated job to the output JSONL file
#     with open(output_file, "a") as f:
#         f.write(json.dumps(job) + "\n")

def process_pdf_to_png(job, output_file):
    pdf_base64 = job['pdf']
    # job_id = job['job_id']

    # Decode the base64 string to binary PDF data
    pdf_data = base64.b64decode(pdf_base64)

    # Load the PDF from binary data
    pdf_document = fitz.open("pdf", pdf_data)

    # Render the first page into a PNG image at 300 DPI
    page = pdf_document.load_page(0)  # Assuming the PDF only has one page
    
    # Initial DPI
    dpi = 300
    zoom = dpi / 72  # 默认dpi是72
    # Render the page at the initial DPI
    # pix = page.get_pixmap(dpi=dpi)
    
    matrix = fitz.Matrix(zoom, zoom)
    # 渲染页面到图像
    pix = page.get_pixmap(matrix=matrix)
    
    # Get the width and height of the image
    width, height = pix.width, pix.height
    
    print("pre", width, height)
    
    # Check if width or height is less than 1500
    if width < 1500 or height < 1500:
        # Calculate the new DPI
        new_dpi = int(1500 * 1500 / (width * height) * 300)
        zoom = new_dpi / 72  # 默认dpi是72
        # Render the page at the new DPI
        matrix = fitz.Matrix(zoom, zoom)
        # 渲染页面到图像
        pix = page.get_pixmap(matrix=matrix)

        width, height = pix.width, pix.height
        
        print("post", width, height)
        
    png_data = pix.tobytes("png")

    # Encode the PNG data to base64
    png_base64 = base64.b64encode(png_data).decode('utf-8')

    # Update the job with the base64 encoded PNG
    job['image'] = png_base64

    # Write the updated job to the output JSONL file
    with open(output_file, "a") as f:
        f.write(json.dumps(job) + "\n")


def process_chunk(process_id, world_size, input_file, output_file):
    err_cnt = 0
    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            if i % world_size == process_id:
                print(f"processing {i}")
                try:
                    job = json.loads(line)
                    # print(job.keys())
                    process_pdf_to_png(job, output_file)
                except:
                    err_cnt += 1
                    print(f"error {err_cnt}")
    
    print(f"final error {err_cnt}")

def main(input_jsonl):
    input_path = Path(input_jsonl)
    output_dir = input_path.stem + "_rendered"
    os.makedirs(output_dir, exist_ok=True)

    world_size = 40  # Number of processes

    # Process the chunks concurrently with multiprocessing
    with multiprocessing.Pool(world_size) as pool:
        pool.starmap(
            process_chunk, 
            [(process_id, world_size, input_jsonl, os.path.join(output_dir, f"{input_path.stem}_output_{process_id}.jsonl")) for process_id in range(world_size)]
        )

if __name__ == "__main__":
    # input_jsonl = "input.jsonl"
    # main(input_jsonl)
    
    if os.path.exists('./train_pdf_with_metadata.jsonl'):
        main('./train_pdf_with_metadata.jsonl')
    
    if os.path.exists('./test_pdf_with_metadata.jsonl'):
        main('./test_pdf_with_metadata.jsonl')
    
    if os.path.exists('./val_pdf_with_metadata.jsonl'):
        main('./val_pdf_with_metadata.jsonl')
        
    if os.path.exists('./combined_pdf.jsonl'):
        main('./combined_pdf.jsonl')
    
