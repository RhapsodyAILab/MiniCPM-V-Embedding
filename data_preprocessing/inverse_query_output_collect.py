import json
import os
import random
import fitz  # PyMuPDF
import re
from PIL import Image

# Set the directory
directory = '/data/K12Textbook_visual_train_inv_query_output'

directory = '/data/icml_sampled_train_inv_query_output'

directory = '/data/manuallib_sampled_train_inv_query_output'

directory = '/data/anna_visual_val_inv_query_output'

output_pdf_filename = directory.split('/')[-1]
num_samples = 50

# Randomly select 50 txt files
files = [f for f in os.listdir(directory) if f.endswith('.txt')]
selected_files = random.sample(files, num_samples)

# Create PDF document
doc = fitz.open()

def parse_text(input_string):
    pattern = r"```json\n(.*?)\n```"
    match = re.search(pattern, input_string, re.DOTALL)
    if not match:
        return None
    return match.group(1)

def manual_wrap(text, char_limit):
    words = text.split()
    lines = []
    current_line = ''
    for word in words:
        if len(current_line) + len(word) + 1 <= char_limit:
            current_line += ' ' + word
        else:
            lines.append(current_line.strip())
            current_line = word
    if current_line:
        lines.append(current_line.strip())
    return lines

for txt_file in selected_files:
    png_file = txt_file.replace('.txt', '.txt.png')
    full_png_path = os.path.join(directory, png_file)
    
    # Adjust image size
    with Image.open(full_png_path) as img:
        img = img.resize((595, 842), Image.LANCZOS)  # Resize to A4 size
        img.save(full_png_path, quality=85)  # Save the image, quality 85

    # Read and parse JSON data
    with open(os.path.join(directory, txt_file), 'r') as file:
        string = json.loads(file.read())
        json_raw_string = parse_text(string['response'])
        data = json.loads(json_raw_string)
    
    # Create a new page and add an image
    img_page = doc.new_page()
    img_rect = fitz.Rect(0, 0, 595, 842)  # A4 paper size
    img_page.insert_image(img_rect, filename=full_png_path)

    # Create another new page and add questions and answers
    text_page = doc.new_page()
    y = 72  # Start from a margin from the top of the page
    line_height = 12

    for item in data['result']:
        question = f"Q: {item['query']}"
        answer = f"A: {item['answer']}"
        for line in manual_wrap(question, 90):
            text_page.insert_text((72, y), line, fontsize=12, fontname="helv")
            y += line_height
        y += 10  # Space between question and answer
        for line in manual_wrap(answer, 90):
            text_page.insert_text((72, y), line, fontsize=12, fontname="helv")
            y += line_height
        y += 20  # Extra space between sets of Q&A

# Save and optimize the PDF
doc.save(f"{output_pdf_filename}_sampled_{num_samples}_optimized.pdf", deflate=True)
doc.close()
