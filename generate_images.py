import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import random
from typing import Tuple
from tqdm import tqdm

class ImageGenerator:
    def __init__(self, output_dir: str = 'dataset'):
        self.output_dir = output_dir
        self.create_directory(output_dir)
        
        # Extended background colors
        self.white_shades = [
            (255, 255, 255),  # Pure white
            (250, 250, 250),  # Very light grayish white
            (240, 240, 240),  # Light grayish white
            (230, 230, 230),  # Lighter shade of gray
            (245, 245, 245),  # Another shade of light gray
            (248, 246, 242),  # Warm white
            (242, 245, 248),  # Cool white
            (252, 251, 248)   # Eggshell white
        ]
        
        # Extended text colors
        self.black_ranges = [
            ((0, 0, 0), (20, 20, 20)),      # Pure to near-black
            ((25, 25, 25), (50, 50, 50)),    # Dark gray
            ((20, 15, 10), (40, 35, 30)),    # Warm dark
            ((10, 10, 20), (30, 30, 40)),    # Cool dark
            ((45, 35, 35), (60, 50, 50))     # Softer dark
        ]
        
        # Enhanced noise parameters
        self.noise_params = {
            'gaussian': {'mean': 0, 'std': 15},
            'speckle': {'mean': 0, 'std': 0.05},
            'salt_and_pepper': {'amount': 0.01}
        }

    # [Previous helper methods remain the same until add_noise]

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add various types of noise to image"""
        noise_type = random.choice(['gaussian', 'speckle', 'salt_and_pepper'])
        
        if noise_type == 'gaussian':
            noise = np.random.normal(self.noise_params['gaussian']['mean'],
                                   self.noise_params['gaussian']['std'],
                                   image.shape).astype(np.uint8)
            noisy = cv2.add(image, noise)
        elif noise_type == 'speckle':
            noise = np.random.normal(self.noise_params['speckle']['mean'],
                                   self.noise_params['speckle']['std'],
                                   image.shape)
            noisy = image + image * noise
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        else:  # salt_and_pepper
            noisy = image.copy()
            amount = self.noise_params['salt_and_pepper']['amount']
            num_salt = np.ceil(amount * image.size * 0.5)
            num_pepper = np.ceil(amount * image.size * 0.5)
            
            # Salt
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            noisy[coords[0], coords[1]] = 255
            
            # Pepper
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            noisy[coords[0], coords[1]] = 0
            
        return noisy

    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply various types of blur to simulate different scanning conditions"""
        blur_types = [
            lambda img: cv2.GaussianBlur(img, (5, 5), random.uniform(0.1, 0.5)),
            lambda img: cv2.blur(img, (3, 3)),
            lambda img: cv2.medianBlur(img, 3),
            lambda img: cv2.GaussianBlur(img, (3, 3), random.uniform(0.1, 0.3)),
            lambda img: cv2.bilateralFilter(img, 5, 75, 75)
        ]
        return random.choice(blur_types)(image)

    def adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Adjust brightness and contrast of the image"""
        # Convert to PIL Image for easier manipulation
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Randomly adjust brightness
        brightness_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness_factor)
        
        # Randomly adjust contrast
        contrast_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def add_jpeg_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Add JPEG compression artifacts"""
        quality = random.randint(60, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def process_word_file(self, input_file: str, augmentations_per_word: int = 14) -> None:
        """Process input file and generate augmented images with varied transformations"""
        with open(input_file, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()

        for i, word in tqdm(enumerate(words), desc="Processing words", total=len(words)):
            for j in range(augmentations_per_word):
                base_image = self.text_to_image(word)
                augmented = base_image
                
                # Apply different combinations of augmentations based on the image index
                if j < 3:  # First 3 images: minimal augmentation
                    if random.random() < 0.5:
                        augmented = self.add_noise(augmented)
                
                elif j < 7:  # Next 4 images: moderate augmentation
                    if random.random() < 0.7:
                        angle = random.uniform(-5, 5)
                        augmented = self.rotate_image(augmented, angle)
                    if random.random() < 0.7:
                        augmented = self.shear_image(augmented)
                    if random.random() < 0.5:
                        augmented = self.adjust_brightness_contrast(augmented)
                
                else:  # Remaining images: heavy augmentation
                    # Apply random combinations of all available augmentations
                    augmentations = [
                        (0.6, lambda: self.rotate_image(augmented, random.uniform(-7, 7))),
                        (0.6, lambda: self.shear_image(augmented)),
                        (0.4, lambda: self.add_noise(augmented)),
                        (0.4, lambda: self.apply_blur(augmented)),
                        (0.4, lambda: self.adjust_brightness_contrast(augmented)),
                        (0.3, lambda: self.add_jpeg_artifacts(augmented))
                    ]
                    
                    for probability, aug_func in augmentations:
                        if random.random() < probability:
                            augmented = aug_func()

                output_path = os.path.join(self.output_dir, f'word_{i}_aug_{j}.png')
                cv2.imwrite(output_path, augmented)

def main():
    input_file = "kaithi_12k.txt"
    generator = ImageGenerator(output_dir='dataset')
    generator.process_word_file(input_file, augmentations_per_word=20)

if __name__ == "__main__":
    main()