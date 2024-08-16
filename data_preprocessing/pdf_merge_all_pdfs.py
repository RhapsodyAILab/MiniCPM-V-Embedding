import os
import base64
# from pypdf import PdfReader, PdfWriter
import json
import sys
import fitz

# def pdf_page_to_base64(pdf_writer):
#     with open("temp.pdf", "wb") as temp_pdf:
#         pdf_writer.write(temp_pdf)
#     with open("temp.pdf", "rb") as temp_pdf:
#         return base64.b64encode(temp_pdf.read()).decode("utf-8")

# def process_pdf(file_path, dataset):
    
#     print(f" ========= processing {file_path} ==========")
#     reader = PdfReader(file_path)
    
#     pdf_base64_list = []
#     for page_num in range(len(reader.pages)):
#         try:
#             writer = PdfWriter()
#             writer.add_page(reader.pages[page_num])
#             base64_pdf = pdf_page_to_base64(writer)
#             pdf_base64_list.append({
#                 "dataset": dataset,
#                 "source": os.path.basename(file_path),
#                 "doc_id": os.path.basename(file_path)+'_'+str(page_num),
#                 "pdf": base64_pdf
#             })
#         except Exception as e:
#             print(f"page error: {e}, skipping")
#     return pdf_base64_list


def pdf_page_to_base64(doc, page_num):
    # 创建新的 PDF 并插入该页
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num, links=False)
    
    # 将新 PDF 保存到字节流
    pdf_bytes = new_doc.write()
    
    # 转换为 base64 字符串
    pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
    return pdf_base64

def process_pdf(file_path, dataset):
    doc = fitz.open(file_path)
    pdf_base64_list = []
    for page_num in range(len(doc)):
        pdf_base64 = pdf_page_to_base64(doc, page_num)
        pdf_base64_list.append({
            "dataset": dataset,
            "source": os.path.basename(file_path),
            "doc_id": os.path.basename(file_path)+'_'+str(page_num),
            "pdf": pdf_base64
        })
    return pdf_base64_list

def main(directory, dataset):
    print("output to", directory.split('/')[-1]+"_pdf_with_metadata.jsonl")
    all_pdfs_base64 = []
    for filename in os.listdir(directory):
        
        print(f"processing {filename}")
        
        try:
            if filename.endswith(".pdf"):
                file_path = os.path.join(directory, filename)
                pdf_base64_list = process_pdf(file_path, dataset)
                all_pdfs_base64.extend(pdf_base64_list)
        except:
            print(f"{filename} exception, skipped")
    
    with open(directory.split('/')[-1]+"_pdf_with_metadata.jsonl", "w") as jsonl_file:
        for pdf_info in all_pdfs_base64:
            jsonl_file.write(json.dumps(pdf_info) + "\n")

if __name__ == "__main__":
    
    dataset = os.getcwd().split('/')[-1]
    print(f"dataset = {dataset}")
    
    # input('>')
    
    if os.path.exists('./train'):
        print("processing train")
        main(directory='./train', dataset=dataset)
    
    if os.path.exists('./test'):
        print("processing test")
        main(directory='./test', dataset=dataset)

    if os.path.exists('./val'):
        print("processing val")
        main(directory='./val', dataset=dataset)