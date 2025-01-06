import os
import shutil
import random

def create_validation_set(dataset_path, validation_path, validation_percentage=0.2):
    """
    Creates a validation set from a dataset of images.

    Args:
        dataset_path: Path to the dataset folder.
        validation_path: Path to the output validation folder.
        validation_percentage: Percentage of images to move to the validation set (default: 20%).
    """

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    if os.path.exists(validation_path):
        print(f"Warning: Validation path '{validation_path}' already exists. Files might be overwritten.")
        # Option to clear the folder if needed:
        # shutil.rmtree(validation_path)

    os.makedirs(validation_path, exist_ok=True)  # Create the validation folder if it doesn't exist

    image_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    num_validation_images = int(len(image_files) * validation_percentage)

    if num_validation_images == 0:
        print("Not enough images in the dataset to create a validation set with the given percentage.")
        return

    validation_indices = random.sample(range(len(image_files)), num_validation_images)

    for i in validation_indices:
        image_name = image_files[i]
        source_path = os.path.join(dataset_path, image_name)
        destination_path = os.path.join(validation_path, image_name)
        try:
            shutil.move(source_path, destination_path) # Move to avoid duplication
            print(f"Moved {image_name} to validation set.")
        except Exception as e:
            print(f"Error moving {image_name}: {e}")

    print(f"Created validation set with {num_validation_images} images in '{validation_path}'.")


if __name__ == "__main__":
    dataset_folder = "dataset"  # Replace with the actual path to your dataset folder
    validation_folder = "kthi_val"
    create_validation_set(dataset_folder, validation_folder)