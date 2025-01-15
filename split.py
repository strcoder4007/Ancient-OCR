import os
import shutil
import random
import csv
import glob
import re
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

def create_validation_set(dataset_path: str, labels_file: str, val_augmentations: int = 5):
    """
    Splits augmentations of each word between training and validation sets.
    
    Args:
        dataset_path: Path to the dataset folder containing images
        labels_file: Path to the input labels CSV file
        val_augmentations: Number of augmentations per word to use for validation (default: 5)
    """
    # Verify dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    # Define and create output directories
    train_folder = os.path.join(dataset_path, 'kthi_train_filtered')
    val_folder = os.path.join(dataset_path, 'kthi_val')

    # Create directories if they don't exist
    for folder in [train_folder, val_folder]:
        if os.path.exists(folder):
            # Clear existing directory
            shutil.rmtree(folder)
        os.makedirs(folder)
        print(f"Created directory: {folder}")

    # Get all image files and group them by word
    image_pattern = os.path.join(dataset_path, 'word_*_aug_*.png')
    all_images = glob.glob(image_pattern)
    
    if not all_images:
        print(f"Error: No images found matching pattern in {dataset_path}")
        return
    
    print(f"Found {len(all_images)} total images")

    # Group images by their word number
    word_groups: Dict[int, List[Tuple[str, int]]] = {}
    for image_path in tqdm(all_images, desc="Grouping images"):
        filename = os.path.basename(image_path)
        match = re.match(r'word_(\d+)_aug_(\d+)\.png', filename)
        if match:
            word_num = int(match.group(1))
            aug_num = int(match.group(2))
            if word_num not in word_groups:
                word_groups[word_num] = []
            word_groups[word_num].append((filename, aug_num))

    print(f"Found {len(word_groups)} unique words")

    # Load labels
    labels_dict = {}
    if not os.path.exists(labels_file):
        print(f"Error: Labels file '{labels_file}' does not exist.")
        return

    with open(labels_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels_dict[row['filename']] = row['words']

    print(f"Loaded {len(labels_dict)} labels from {labels_file}")

    validation_files = []
    training_files = []

    # For each word, randomly select augmentations for validation
    for word_num, files_and_augs in tqdm(word_groups.items(), desc="Splitting datasets"):
        # Sort by augmentation number to ensure consistent ordering
        files_and_augs.sort(key=lambda x: x[1])
        
        # Select random augmentations for validation
        total_augs = len(files_and_augs)
        val_indices = set(random.sample(range(total_augs), val_augmentations))
        
        # Split files between training and validation
        for idx, (filename, _) in enumerate(files_and_augs):
            if idx in val_indices:
                validation_files.append(filename)
            else:
                training_files.append(filename)

    def move_files_and_create_labels(file_list: List[str], destination_folder: str, 
                                   labels_output_file: str) -> None:
        """
        Move files to destination and create corresponding labels file.
        """
        successful_moves = 0
        failed_moves = 0
        
        with open(labels_output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'words'])
            
            for filename in tqdm(file_list, desc=f"Moving files to {os.path.basename(destination_folder)}"):
                source_path = os.path.join(dataset_path, filename)
                destination_path = os.path.join(destination_folder, filename)
                
                try:
                    if os.path.exists(source_path):
                        shutil.move(source_path, destination_path)  # Changed from copy2 to move
                        successful_moves += 1
                        
                        # Get the base filename to find the label
                        base_filename = re.sub(r'aug_\d+', 'aug_0', filename)
                        if base_filename in labels_dict:
                            writer.writerow([filename, labels_dict[base_filename]])
                        else:
                            print(f"Warning: No label found for {filename}")
                            failed_moves += 1
                    else:
                        print(f"Warning: {filename} not found in dataset path")
                        failed_moves += 1
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    failed_moves += 1

        return successful_moves, failed_moves

    # Move files and create labels for both sets
    print("\nProcessing validation set...")
    val_success, val_failed = move_files_and_create_labels(
        validation_files,
        val_folder,
        os.path.join(val_folder, 'labels.csv')
    )

    print("\nProcessing training set...")
    train_success, train_failed = move_files_and_create_labels(
        training_files,
        train_folder,
        os.path.join(train_folder, 'labels.csv')
    )

    # Print detailed summary
    print(f"\nProcessing Complete!")
    print(f"Training Set:")
    print(f"  - Successful moves: {train_success}")
    print(f"  - Failed moves: {train_failed}")
    print(f"  - Total files: {len(training_files)}")
    print(f"  - Directory: {train_folder}")
    
    print(f"\nValidation Set:")
    print(f"  - Successful moves: {val_success}")
    print(f"  - Failed moves: {val_failed}")
    print(f"  - Total files: {len(validation_files)}")
    print(f"  - Directory: {val_folder}")
    
    print(f"\nSplit Ratio:")
    total_files = len(training_files) + len(validation_files)
    print(f"  - Training: {len(training_files)/total_files:.1%}")
    print(f"  - Validation: {len(validation_files)/total_files:.1%}")
    print(f"  - {20-val_augmentations} augmentations per word in training")
    print(f"  - {val_augmentations} augmentations per word in validation")

if __name__ == "__main__":
    dataset_folder = "all_data"
    labels_csv_file = "labels.csv"
    validation_augmentations = 3
    
    create_validation_set(dataset_folder, labels_csv_file, validation_augmentations)