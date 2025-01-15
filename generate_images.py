import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import random
from typing import Tuple, Optional
from tqdm import tqdm

class ImageGenerator:
    def __init__(self, output_dir: str = 'all_data', min_zoom: float = 0.8, max_zoom: float = 1.2):
        self.output_dir = output_dir
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.white_shades = [
            (255, 255, 255),  
            (250, 250, 250),  
            (240, 240, 240),  
            (245, 245, 245),  
            (248, 246, 242),  
            (252, 251, 248),  
            (253, 253, 250)  
        ]
        
        self.black_ranges = [
            ((0, 0, 0), (20, 20, 20)),   
            ((25, 25, 25), (50, 50, 50)), 
            ((20, 15, 10), (40, 35, 30)),
            ((10, 10, 20), (30, 30, 40)),
            ((15, 15, 25), (35, 35, 45))
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
        padding_x = int(text_width * 0)
        padding_y = int(text_height * 0)
        
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
        shear_factor = random.uniform(-0.4, 0.4)
        
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
        noise_type = random.choice(['gaussian', 'speckle', 'salt_and_pepper'])
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, random.uniform(5, 15), image.shape).astype(np.uint8)
            noisy = cv2.add(image, noise)
        elif noise_type == 'speckle':
            noise = np.random.normal(0, random.uniform(0.02, 0.05), image.shape)
            noisy = image + image * noise
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        else:
            s_vs_p = 0.5
            amount = random.uniform(0.02, 0.1)
            noisy = image.copy()
            total_pixels = image.size
            num_salt = int(total_pixels * amount * s_vs_p)
            num_pepper = int(total_pixels * amount * (1.0 - s_vs_p))

            salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
            noisy[salt_coords[0], salt_coords[1]] = 255

            pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
            noisy[pepper_coords[0], pepper_coords[1]] = 0
        
        return noisy

    def adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        brightness_factor = random.uniform(0.7, 1.5)
        contrast_factor = random.uniform(0.7, 1.5)
        
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

    def apply_zoom(self, image: np.ndarray, zoom_factor: Optional[float] = None) -> np.ndarray:
        """Apply zoom augmentation to the image."""
        if zoom_factor is None:
            zoom_factor = random.uniform(self.min_zoom, self.max_zoom)
        
        height, width = image.shape[:2]
        
        new_height = int(height * zoom_factor)
        new_width = int(width * zoom_factor)
        
        zoomed = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        if zoom_factor < 1:
            pad_height = height - new_height
            pad_width = width - new_width
            
            top_pad = pad_height // 2
            bottom_pad = pad_height - top_pad
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            
            bg_color = self.get_random_white_shade()
            zoomed = cv2.copyMakeBorder(zoomed, top_pad, bottom_pad, left_pad, right_pad,
                                      cv2.BORDER_CONSTANT, value=bg_color)
        elif zoom_factor > 1:
            start_y = (new_height - height) // 2
            start_x = (new_width - width) // 2
            zoomed = zoomed[start_y:start_y + height, start_x:start_x + width]
        
        return zoomed

    def apply_perspective_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply random perspective transformation."""
        height, width = image.shape[:2]
        margin = min(width, height) * 0.1
        src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        dst_points = np.float32([
            [random.uniform(0, margin), random.uniform(0, margin)],
            [random.uniform(width - margin, width), random.uniform(0, margin)],
            [random.uniform(0, margin), random.uniform(height - margin, height)],
            [random.uniform(width - margin, width), random.uniform(height - margin, height)]
        ])
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        bg_color = self.get_random_white_shade()
        
        # Apply perspective transform
        return cv2.warpPerspective(image, matrix, (width, height),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=bg_color)

    def process_word_file(self, input_file: str, augmentations_per_word: int = 20) -> None:
        with open(input_file, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()

        for i, word in tqdm(enumerate(words), desc="Processing words", total=len(words)):
            for j in range(augmentations_per_word):
                # Create base image
                base_image = self.text_to_image(word)
                augmented = base_image.copy()
                if j < 5:
                    if random.random() < 0.3:
                        augmented = self.adjust_brightness_contrast(augmented)
                    if random.random() < 0.2:
                        augmented = self.apply_zoom(augmented, random.uniform(0.95, 1.05))
                elif j < 10:
                    if random.random() < 0.8:
                        angle = random.uniform(-5, 5)
                        augmented = self.rotate_image(augmented, angle)
                    if random.random() < 0.6:
                        augmented = self.adjust_brightness_contrast(augmented)
                    if random.random() < 0.4:
                        augmented = self.apply_zoom(augmented, random.uniform(0.9, 1.1))
                elif j < 15:
                    if random.random() < 0.7:
                        angle = random.uniform(-7, 7)
                        augmented = self.rotate_image(augmented, angle)
                    if random.random() < 0.8:
                        augmented = self.shear_image(augmented)
                    if random.random() < 0.6:
                        augmented = self.add_noise(augmented)
                    if random.random() < 0.6:
                        augmented = self.apply_zoom(augmented, random.uniform(0.85, 1.15))
                    if random.random() < 0.4:
                        augmented = self.apply_perspective_transform(augmented)
                else:
                    if random.random() < 0.9:
                        angle = random.uniform(-10, 10)
                        augmented = self.rotate_image(augmented, angle)
                    if random.random() < 0.9:
                        augmented = self.shear_image(augmented)
                    if random.random() < 0.7:
                        augmented = self.add_noise(augmented)
                    if random.random() < 0.7:
                        augmented = self.apply_blur(augmented)
                    if random.random() < 0.7:
                        augmented = self.add_noise(augmented)
                    if random.random() < 0.7:
                        augmented = self.apply_blur(augmented)
                    if random.random() < 0.8:
                        augmented = self.adjust_brightness_contrast(augmented)
                    if random.random() < 0.8:
                        augmented = self.apply_zoom(augmented, random.uniform(0.8, 1.2))
                    if random.random() < 0.6:
                        augmented = self.apply_perspective_transform(augmented)

                if augmented.shape[:2] != base_image.shape[:2]:
                    augmented = cv2.resize(augmented, (base_image.shape[1], base_image.shape[0]))
                
                output_path = os.path.join(self.output_dir, f'word_{i}_aug_{j}.png')
                cv2.imwrite(output_path, augmented)

def main():
    input_file = "kaithi_10.txt"
    generator = ImageGenerator(output_dir='all_data', min_zoom=0.8, max_zoom=1.2)
    generator.process_word_file(input_file, augmentations_per_word=20)

if __name__ == "__main__":
    main()