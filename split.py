import os
import shutil
import random
import csv

def create_validation_set(dataset_path, labels_file, validation_percentage=0.2):
    """
    Creates a validation set from a dataset of images and the corresponding labels.

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

    # Step 2: Load the labels from the CSV file
    image_files = []
    labels = []

    if not os.path.exists(labels_file):
        print(f"Error: Labels file '{labels_file}' does not exist.")
        return

    with open(labels_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_files.append(row['filename'])
            labels.append(row['words'])

    num_validation_images = int(len(image_files) * validation_percentage)

    if num_validation_images == 0:
        print("Not enough images in the dataset to create a validation set with the given percentage.")
        return

    # Step 3: Select random validation images
    validation_indices = random.sample(range(len(image_files)), num_validation_images)

    validation_image_files = [image_files[i] for i in validation_indices]
    validation_labels = [labels[i] for i in validation_indices]

    # Step 4: Move images to the validation folder and update CSV entries
    for i in validation_indices:
        image_name = image_files[i]
        source_path = os.path.join(dataset_path, image_name)
        destination_path = os.path.join(val_folder, image_name)
        
        # Move image to validation folder
        try:
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
                print(f"Moved {image_name} to validation set.")
            else:
                print(f"Warning: {image_name} not found in dataset path.")
        except Exception as e:
            print(f"Error moving {image_name}: {e}")

    # Step 5: Move remaining images to training folder and update CSV entries
    train_image_files = [image_files[i] for i in range(len(image_files)) if i not in validation_indices]
    train_labels = [labels[i] for i in range(len(labels)) if i not in validation_indices]

    for image_name in train_image_files:
        source_path = os.path.join(dataset_path, image_name)
        destination_path = os.path.join(train_folder, image_name)
        
        # Move image to training folder
        try:
            if os.path.exists(source_path):
                if not os.path.exists(destination_path):  # Avoid overwriting existing files
                    shutil.move(source_path, destination_path)
                    print(f"Moved {image_name} to training set.")
                else:
                    print(f"Warning: {image_name} already exists in training set, skipping.")
            else:
                print(f"Warning: {image_name} not found in dataset path.")
        except Exception as e:
            print(f"Error moving {image_name}: {e}")

    # Step 6: Create new CSV files for train and validation sets
    with open(os.path.join(val_folder, 'labels.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'words'])  # Write header
        for image_name, label in zip(validation_image_files, validation_labels):
            writer.writerow([image_name, label])

    with open(os.path.join(train_folder, 'labels.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'words'])  # Write header
        for image_name, label in zip(train_image_files, train_labels):
            writer.writerow([image_name, label])

    print(f"Created training set with {len(train_image_files)} images in '{train_folder}' and labels in 'train_labels.csv'.")
    print(f"Created validation set with {num_validation_images} images in '{val_folder}' and labels in 'validation_labels.csv'.")

if __name__ == "__main__":
    dataset_folder = "dataset"  # Path to your dataset folder
    labels_csv_file = "labels.csv"  # Path to the input labels.csv

    create_validation_set(dataset_folder, labels_csv_file)
