import fitz  # PyMuPDF
import os
import shutil
from langdetect import detect
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_text_and_detect_language(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page in doc:
            try:
                text += page.get_text()
            except:
                print("read page error")
                pass
        
        language = detect(text)
        return pdf_path, language
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return pdf_path, None

def copy_if_english(result):
    pdf_path, language = result
    if language == 'en':
        shutil.copy(pdf_path, dest_dir)
        print(f"Copied: {os.path.basename(pdf_path)}")

def process_pdfs(source_dir):
    pdf_files = [os.path.join(source_dir, filename) for filename in os.listdir(source_dir) if filename.endswith('.pdf')]
    with ProcessPoolExecutor(max_workers=64) as executor:
        future_to_pdf = {executor.submit(extract_text_and_detect_language, pdf): pdf for pdf in pdf_files}
        for future in as_completed(future_to_pdf):
            try:
                result = future.result()
                copy_if_english(result)
            except Exception as exc:
                print(f"Generated an exception: {exc}")




source_directory = './all_pdfs'  # 替换为你的源目录路径
dest_dir = './original'  # 替换为你的目标目录路径

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

process_pdfs(source_directory)
