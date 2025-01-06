import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import random
from typing import Tuple
from tqdm import tqdm

class ImageGenerator:
    def __init__(self, output_dir: str = 'dataset', image_size: Tuple[int, int] = (400, 100)):
        """Initialize the image generator with configuration parameters"""
        self.output_dir = output_dir
        self.width, self.height = image_size
        self.create_directory(output_dir)
        
        # Color ranges for scanned paper/scriptures (only yellows and white for background)
        self.yellow_ranges = [
            ((255, 255, 220), (255, 255, 240)),  
            ((255, 253, 208), (255, 253, 225)),  
            ((255, 250, 190), (255, 250, 210)), 
            ((255, 245, 170), (255, 245, 190)), 
            ((255, 255, 255), (255, 255, 255))
        ]
        
        self.black_ranges = [
            ((0, 0, 0), (20, 20, 20)),
            ((25, 25, 25), (50, 50, 50)),
            ((20, 15, 10), (40, 35, 30)),
            ((10, 10, 20), (30, 30, 40)) 
        ]
        
        # Noise parameters (for more faded or dirty textures)
        self.noise_params = {
            'gaussian': {'mean': 0, 'std': 15},  # Reduced to make the noise subtler
            'speckle': {'mean': 0, 'std': 0.05}  # Slight speckle noise to simulate dirt
        }

    @staticmethod
    def create_directory(dir_name: str) -> None:
        """Create directory if it doesn't exist"""
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def get_random_color(self, ranges: list) -> Tuple[int, int, int]:
        """Get a random color within specified ranges"""
        range_choice = random.choice(ranges)
        return tuple(random.randint(min_val, max_val) 
                    for min_val, max_val in zip(range_choice[0], range_choice[1]))

    def text_to_image(self, text: str, font_size: int = 48) -> np.ndarray:
        """Convert text to image with random background and foreground colors"""
        # Create a new image with a yellow or white background (only yellow and white)
        image = Image.new('RGB', (self.width, self.height), self.get_random_color(self.yellow_ranges))
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansKaithi-Regular.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Get text size and position
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (self.width - text_width) // 2
        y = (self.height - text_height) // 2
        
        # Draw text with a random dark color resembling old ink
        draw.text((x, y), text, font=font, fill=self.get_random_color(self.black_ranges))
        
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def rotate_image(self, image: np.ndarray) -> np.ndarray:
        """Rotate image by random angle to simulate paper bending"""
        angle = random.uniform(-5, 5)  # Slight rotation
        center = (self.width // 2, self.height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        bg_color = self.get_random_color(self.yellow_ranges)
        return cv2.warpAffine(image, rotation_matrix, (self.width, self.height), 
                            borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)

    def shear_image(self, image: np.ndarray) -> np.ndarray:
        """Apply random shear transformation"""
        shear_factor = random.uniform(-0.2, 0.2)  # Slight shear
        pts1 = np.float32([[0, 0], [self.width, 0], [0, self.height]])
        pts2 = np.float32([[0, 0], [self.width, 0], 
                          [int(shear_factor * self.width), self.height]])
        shear_matrix = cv2.getAffineTransform(pts1, pts2)
        bg_color = self.get_random_color(self.yellow_ranges)
        return cv2.warpAffine(image, shear_matrix, (self.width, self.height), 
                            borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add random noise to image to simulate dirt or scanner imperfections"""
        if random.choice([True, False]):
            # Gaussian noise (faded look)
            noise = np.random.normal(self.noise_params['gaussian']['mean'], 
                                   self.noise_params['gaussian']['std'], 
                                   image.shape).astype(np.uint8)
            noisy = cv2.add(image, noise)
        else:
            # Speckle noise (dirt)
            noise = np.random.normal(self.noise_params['speckle']['mean'], 
                                   self.noise_params['speckle']['std'], 
                                   image.shape)
            noisy = image + image * noise
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle blur to simulate scanner imperfections"""
        blur_types = [
            lambda img: cv2.GaussianBlur(img, (5, 5), random.uniform(0.1, 0.3)),
            lambda img: cv2.blur(img, (5, 5)),
            lambda img: cv2.medianBlur(img, 5) 
        ]
        return random.choice(blur_types)(image)

    def process_word_file(self, input_file: str, augmentations_per_word: int = 10) -> None:
        """Process input file and generate augmented images"""
        with open(input_file, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()

        for i, word in tqdm(enumerate(words), desc="Processing words", total=len(words)):
            for j in tqdm(range(augmentations_per_word), desc=f"Generating images for word {word}", leave=False):
                base_image = self.text_to_image(word)
                augmented = base_image
                
                # Randomly apply transformations
                if random.random() < 0.7:
                    augmented = self.rotate_image(augmented)
                if random.random() < 0.7:
                    augmented = self.shear_image(augmented)
                if random.random() < 0.4:
                    augmented = self.add_noise(augmented)
                if random.random() < 0.3:
                    augmented = self.apply_blur(augmented)
                
                # Save augmented image
                output_path = os.path.join(self.output_dir, f'word_{i}_aug_{j}.png')
                cv2.imwrite(output_path, augmented)

def main():
    input_file = "kaithi_1000.txt"
    generator = ImageGenerator(output_dir='dataset', image_size=(400, 100))
    generator.process_word_file(input_file, augmentations_per_word=10)

if __name__ == "__main__":
    main()
