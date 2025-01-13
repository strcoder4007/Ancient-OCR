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
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.white_shades = [
            (255, 255, 255),  # Pure white
            (250, 250, 250),  # Very light grayish white
            (240, 240, 240),  # Light grayish white
            (245, 245, 245),  # Another shade of light gray
            (248, 246, 242),  # Warm white
            (252, 251, 248)   # Eggshell white
        ]
        
        self.black_ranges = [
            ((0, 0, 0), (20, 20, 20)),      # Pure to near-black
            ((25, 25, 25), (50, 50, 50)),    # Dark gray
            ((20, 15, 10), (40, 35, 30)),    # Warm dark
            ((10, 10, 20), (30, 30, 40))     # Cool dark
        ]

    def get_random_color(self, ranges: list) -> Tuple[int, int, int]:
        range_choice = random.choice(ranges)
        return tuple(random.randint(min_val, max_val) 
                    for min_val, max_val in zip(range_choice[0], range_choice[1]))

    def get_random_white_shade(self) -> Tuple[int, int, int]:
        return random.choice(self.white_shades)

    def get_random_font(self) -> ImageFont.FreeTypeFont:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        fonts_dir = os.path.join(script_dir, 'fonts')
        
        fonts = [
            os.path.join(fonts_dir, "NotoSansKaithi-Regular.ttf"),
        ]
        font_choice = random.choice(fonts)
        font_size = random.randint(36, 48)
        try:
            return ImageFont.truetype(font_choice, font_size)
        except:
            return ImageFont.load_default()

    def get_text_dimensions(self, text: str, font: ImageFont.FreeTypeFont) -> Tuple[Tuple[int, int, int, int], Tuple[int, int]]:
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        return bbox, (text_width, text_height)

    def calculate_image_size(self, text_width: int, text_height: int) -> Tuple[int, int]:
        padding_x = int(text_width * 0.2)
        padding_y = int(text_height * 0.2)
        
        final_width = text_width + (2 * padding_x)
        final_height = text_height + (2 * padding_y)
        
        return final_width, final_height

    def text_to_image(self, text: str) -> np.ndarray:
        font = self.get_random_font()
        bbox, (text_width, text_height) = self.get_text_dimensions(text, font)
        
        width, height = self.calculate_image_size(text_width, text_height)
        
        image = Image.new('RGB', (width, height), self.get_random_white_shade())
        draw = ImageDraw.Draw(image)
        
        x = (width - text_width) // 2 - bbox[0]
        y = (height - text_height) // 2 - bbox[1]
        
        draw.text((x, y), text, font=font, fill=self.get_random_color(self.black_ranges))
        
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        
        new_width = int(height * abs_sin + width * abs_cos)
        new_height = int(height * abs_cos + width * abs_sin)
        
        rotation_matrix[0, 2] += new_width/2 - center[0]
        rotation_matrix[1, 2] += new_height/2 - center[1]
        
        bg_color = self.get_random_white_shade()
        return cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=bg_color)

    def shear_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
<<<<<<< HEAD
        shear_factor = random.uniform(-0.4, 0.4)  # Increased shear factor for more distortion
=======
        shear_factor = random.uniform(-0.2, 0.2)
>>>>>>> f2db6a7abc6f56665e63cb662b98eeaec0b0286c
        
        new_width = width + int(abs(shear_factor * height))
        x_offset = (new_width - width) // 2
        
        pts1 = np.float32([[0, 0], [width, 0], [0, height]])
        pts2 = np.float32([[x_offset, 0],
                          [x_offset + width, 0],
                          [x_offset + int(shear_factor * height), height]])
        
        shear_matrix = cv2.getAffineTransform(pts1, pts2)
        bg_color = self.get_random_white_shade()
        
        return cv2.warpAffine(image, shear_matrix, (new_width, height),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=bg_color)

    def add_noise(self, image: np.ndarray) -> np.ndarray:
<<<<<<< HEAD
        noise_type = random.choice(['gaussian', 'speckle', 'salt_and_pepper'])
=======
        noise_type = random.choice(['gaussian', 'speckle'])
>>>>>>> f2db6a7abc6f56665e63cb662b98eeaec0b0286c
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, random.uniform(5, 15), image.shape).astype(np.uint8)
            noisy = cv2.add(image, noise)
<<<<<<< HEAD
        elif noise_type == 'speckle':
            noise = np.random.normal(0, random.uniform(0.02, 0.05), image.shape)
            noisy = image + image * noise
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        else:  # salt_and_pepper
            s_vs_p = 0.5
            amount = random.uniform(0.02, 0.1)
            noisy = image.copy()
            total_pixels = image.size
            num_salt = int(total_pixels * amount * s_vs_p)
            num_pepper = int(total_pixels * amount * (1.0 - s_vs_p))
            
            # Salt noise
            salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
            noisy[salt_coords[0], salt_coords[1]] = 255
            
            # Pepper noise
            pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
            noisy[pepper_coords[0], pepper_coords[1]] = 0
=======
        else:  # speckle
            noise = np.random.normal(0, random.uniform(0.02, 0.05), image.shape)
            noisy = image + image * noise
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
>>>>>>> f2db6a7abc6f56665e63cb662b98eeaec0b0286c
        
        return noisy

    def adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
<<<<<<< HEAD
        brightness_factor = random.uniform(0.7, 1.5)  # Increased range for brightness
        contrast_factor = random.uniform(0.7, 1.5)    # Increased range for contrast
=======
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
>>>>>>> f2db6a7abc6f56665e63cb662b98eeaec0b0286c
        
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness_factor)
        
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        blur_types = [
            lambda img: cv2.GaussianBlur(img, (3, 3), random.uniform(0.1, 0.5)),
            lambda img: cv2.blur(img, (3, 3)),
            lambda img: cv2.medianBlur(img, 3)
        ]
        return random.choice(blur_types)(image)

    def process_word_file(self, input_file: str, augmentations_per_word: int = 20) -> None:
        with open(input_file, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()

        for i, word in tqdm(enumerate(words), desc="Processing words", total=len(words)):
            # Generate base images with different fonts/colors
            for j in range(augmentations_per_word):
                # Create base image
                base_image = self.text_to_image(word)
                augmented = base_image.copy()
                
                # Group 1 (0-4): Minimal augmentation (clean samples)
                if j < 5:
                    if random.random() < 0.3:
                        augmented = self.adjust_brightness_contrast(augmented)
                
                # Group 2 (5-9): Light augmentation
                elif j < 10:
<<<<<<< HEAD
                    if random.random() < 0.8:
                        angle = random.uniform(-5, 5)  # Increased rotation range
                        augmented = self.rotate_image(augmented, angle)
                    if random.random() < 0.6:
=======
                    if random.random() < 0.7:
                        angle = random.uniform(-3, 3)
                        augmented = self.rotate_image(augmented, angle)
                    if random.random() < 0.5:
>>>>>>> f2db6a7abc6f56665e63cb662b98eeaec0b0286c
                        augmented = self.adjust_brightness_contrast(augmented)
                
                # Group 3 (10-14): Medium augmentation
                elif j < 15:
                    if random.random() < 0.7:
<<<<<<< HEAD
                        angle = random.uniform(-7, 7)  # Increased rotation range
                        augmented = self.rotate_image(augmented, angle)
                    if random.random() < 0.8:
                        augmented = self.shear_image(augmented)  # Increased shearing
                    if random.random() < 0.6:
=======
                        angle = random.uniform(-5, 5)
                        augmented = self.rotate_image(augmented, angle)
                    if random.random() < 0.7:
                        augmented = self.shear_image(augmented)
                    if random.random() < 0.5:
>>>>>>> f2db6a7abc6f56665e63cb662b98eeaec0b0286c
                        augmented = self.add_noise(augmented)
                
                # Group 4 (15-19): Heavy augmentation
                else:
                    # Apply multiple augmentations
<<<<<<< HEAD
                    if random.random() < 0.9:
                        angle = random.uniform(-10, 10)  # Larger rotation
                        augmented = self.rotate_image(augmented, angle)
                    if random.random() < 0.9:
                        augmented = self.shear_image(augmented)  # Larger shearing
                    if random.random() < 0.7:
                        augmented = self.add_noise(augmented)
                    if random.random() < 0.7:
                        augmented = self.apply_blur(augmented)
                    if random.random() < 0.8:
=======
                    if random.random() < 0.8:
                        angle = random.uniform(-7, 7)
                        augmented = self.rotate_image(augmented, angle)
                    if random.random() < 0.8:
                        augmented = self.shear_image(augmented)
                    if random.random() < 0.6:
                        augmented = self.add_noise(augmented)
                    if random.random() < 0.6:
                        augmented = self.apply_blur(augmented)
                    if random.random() < 0.7:
>>>>>>> f2db6a7abc6f56665e63cb662b98eeaec0b0286c
                        augmented = self.adjust_brightness_contrast(augmented)
                
                output_path = os.path.join(self.output_dir, f'word_{i}_aug_{j}.png')
                cv2.imwrite(output_path, augmented)

def main():
    input_file = "kaithi_12k.txt"
    generator = ImageGenerator(output_dir='dataset')
    generator.process_word_file(input_file, augmentations_per_word=20)

if __name__ == "__main__":
    main()