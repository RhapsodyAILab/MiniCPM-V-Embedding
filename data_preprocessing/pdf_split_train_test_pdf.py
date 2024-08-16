import os
import shutil
import random
from pypdf import PdfReader

# 设置随机种子
random.seed(42)


PDF_DIR = './original'

TRAIN_N_PAGES=20000
TEST_N_PAGES=20000

def count_pdf_pages(pdf_path):
    reader = PdfReader(pdf_path)
    return len(reader.pages)

def get_all_pdfs(directory):
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def split_pdfs(pdf_files, target_page_count_train=20000, target_page_count_test=5000):
    total_pages = sum(pdf[1] for pdf in pdf_files)
    random.shuffle(pdf_files)
    
    split1, split2 = [], []
    pages1, pages2 = 0, 0
    
    for pdf, pages in pdf_files:
        if pages1 < target_page_count_train:
            split1.append(pdf)
            pages1 += pages
        elif pages2 < target_page_count_test:
            split2.append(pdf)
            pages2 += pages
        else:
            break
    
    return split1, split2

def save_pdfs(pdfs, target_directory):
    os.makedirs(target_directory, exist_ok=True)
    for pdf in pdfs:
        shutil.copy(pdf, target_directory)

def main(directory):
    pdf_files = get_all_pdfs(directory)
    
    pdf_files_with_pages = []
    
    for pdf in pdf_files:
        print(f"counting pages {pdf}")
        try:
            num_pages = count_pdf_pages(pdf)
            pdf_files_with_pages.append((pdf, num_pages))
        except:
            pass
        # (pdf, ) 
    
    print("begin to split data")
    
    train_pdfs, test_pdfs = split_pdfs(pdf_files_with_pages, TRAIN_N_PAGES, TEST_N_PAGES)
    
    save_pdfs(train_pdfs, os.path.join('train'))
    save_pdfs(test_pdfs, os.path.join('val'))

if __name__ == '__main__':
    
    main(PDF_DIR)

