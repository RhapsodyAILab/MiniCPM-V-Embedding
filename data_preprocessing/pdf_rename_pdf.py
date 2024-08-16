import os

def rename_pdfs(directory):
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    pdf_files.sort()  # Optional: Sort the files if you want them renamed in a specific order

    for i, filename in enumerate(pdf_files, start=1):
        new_name = f"book_{i}.pdf"
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_name}'")

# 使用示例：将'directory_path'替换为你的目录路径
directory_path = '.'
rename_pdfs(directory_path)