import os
import hashlib

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def remove_duplicate_files(directory):
    md5_dict = {}
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            file_md5 = calculate_md5(file_path)
            if file_md5 in md5_dict:
                print(f"Duplicate found: {file_path} and {md5_dict[file_md5]}")
                os.remove(file_path)
                print(f"Removed: {file_path}")
            else:
                md5_dict[file_md5] = file_path

if __name__ == "__main__":
    directory = "./all_pdfs"
    remove_duplicate_files(directory)
