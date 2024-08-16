import os
import random
import shutil

def split_parquet_files(input_dir, train_dir, test_dir, train_size=150, test_size=50, seed=None):
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Get the list of all parquet files in the input directory
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]

    # Shuffle the list of files
    random.shuffle(parquet_files)

    # Determine the actual number of files to move
    total_files = len(parquet_files)
    actual_train_size = min(train_size, total_files)
    actual_test_size = min(test_size, total_files - actual_train_size)
    
    # Split the files
    train_files = parquet_files[:actual_train_size]
    test_files = parquet_files[actual_train_size:actual_train_size + actual_test_size]

    # Create the train and test directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy the files to the respective directories
    for file in train_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(train_dir, file))
    
    for file in test_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(test_dir, file))

    print(f'Copied {len(train_files)} files to {train_dir}')
    print(f'Copied {len(test_files)} files to {test_dir}')

# Example usage
split_parquet_files(
    input_dir='./original',
    train_dir='./train',
    test_dir='./test',
    seed=42
)
