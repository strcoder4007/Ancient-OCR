import os
import shutil
import random
import csv
import glob
import re

def create_validation_set(dataset_path, labels_file, validation_percentage=0.2):
    """
    Creates a validation set from a dataset of images and the corresponding labels,
    including all augmented images (aug_0 through aug_13).

    Args:
        dataset_path: Path to the dataset folder containing images.
        labels_file: Path to the input labels CSV file containing 'filename' and 'words'.
        validation_percentage: Percentage of images to move to the validation set (default: 20%).
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    # Step 1: Prepare paths for training and validation sets
    train_folder = os.path.join(dataset_path, 'kthi_train_filtered')
    val_folder = os.path.join(dataset_path, 'kthi_val')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Step 2: Get all image files and group them by base word
    image_pattern = os.path.join(dataset_path, 'word_*_aug_*.png')
    all_images = glob.glob(image_pattern)
    
    # Group images by their base word number
    word_groups = {}
    for image_path in all_images:
        filename = os.path.basename(image_path)
        # Extract word number using regex
        match = re.match(r'word_(\d+)_aug_\d+\.png', filename)
        if match:
            word_num = int(match.group(1))
            if word_num not in word_groups:
                word_groups[word_num] = []
            word_groups[word_num].append(filename)

    # Step 3: Load the labels from the CSV file
    labels_dict = {}
    if not os.path.exists(labels_file):
        print(f"Error: Labels file '{labels_file}' does not exist.")
        return

    with open(labels_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels_dict[row['filename']] = row['words']

    # Step 4: Split words into train and validation sets
    word_numbers = list(word_groups.keys())
    num_validation_words = int(len(word_numbers) * validation_percentage)
    validation_word_numbers = set(random.sample(word_numbers, num_validation_words))

    validation_files = []
    training_files = []

    # Distribute all augmented versions based on their word number
    for word_num, files in word_groups.items():
        if word_num in validation_word_numbers:
            validation_files.extend(files)
        else:
            training_files.extend(files)

    # Step 5: Move files and create new label files
    def move_files_and_create_labels(file_list, destination_folder, labels_output_file):
        with open(labels_output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'words'])
            
            for filename in file_list:
                source_path = os.path.join(dataset_path, filename)
                destination_path = os.path.join(destination_folder, filename)
                
                try:
                    if os.path.exists(source_path):
                        if not os.path.exists(destination_path):
                            shutil.move(source_path, destination_path)
                            print(f"Moved {filename} to {os.path.basename(destination_folder)}")
                        else:
                            print(f"Warning: {filename} already exists in destination, skipping.")
                    else:
                        print(f"Warning: {filename} not found in dataset path.")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
                    continue

                # Get the base filename (word_X_aug_0.png) to find the label
                base_filename = re.sub(r'aug_\d+', 'aug_0', filename)
                if base_filename in labels_dict:
                    writer.writerow([filename, labels_dict[base_filename]])
                else:
                    print(f"Warning: No label found for {filename}")

    # Move and create labels for validation set
    move_files_and_create_labels(
        validation_files,
        val_folder,
        os.path.join(val_folder, 'labels.csv')
    )

    # Move and create labels for training set
    move_files_and_create_labels(
        training_files,
        train_folder,
        os.path.join(train_folder, 'labels.csv')
    )

    print(f"\nSummary:")
    print(f"Created training set with {len(training_files)} images in '{train_folder}'")
    print(f"Created validation set with {len(validation_files)} images in '{val_folder}'")
    print(f"Split ratio: {len(training_files)/(len(training_files)+len(validation_files)):.1%} training, "
          f"{len(validation_files)/(len(training_files)+len(validation_files)):.1%} validation")

if __name__ == "__main__":
    dataset_folder = "dataset"  # Path to your dataset folder
    labels_csv_file = "labels.csv"  # Path to the input labels.csv

    create_validation_set(dataset_folder, labels_csv_file)