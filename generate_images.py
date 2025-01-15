import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import random
from tqdm import tqdm
import csv

class ImageGenerator:
    def __init__(self, output_dir: str = 'all_data', min_zoom: float = 0.8, max_zoom: float = 1.2):
        self.output_dir = output_dir
        self.min_zoom, self.max_zoom = min_zoom, max_zoom
        os.makedirs(output_dir, exist_ok=True)
        self.white_shades = [(255, 255, 255), (250, 250, 250)]
        self.black = (0, 0, 0)
        self.labels_file = os.path.join(output_dir, 'labels.csv')
        
        # Initialize CSV file with header (only once)
        with open(self.labels_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'words'])  # Only write header once

        # Standard height for all text images
        self.standard_height = 64

    def get_random_font(self):
        font_path = os.path.join(os.path.dirname(__file__), 'fonts', "NotoSansKaithi-Regular.ttf")
        return ImageFont.truetype(font_path, random.randint(36, 48)) if os.path.exists(font_path) else ImageFont.load_default()

    def get_text_dimensions(self, text, font):
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        bbox = draw.textbbox((0, 0), text, font)
        return bbox, (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def text_to_image(self, text):
        font = self.get_random_font()
        bbox, (text_width, text_height) = self.get_text_dimensions(text, font)
        img = Image.new('RGB', (text_width, text_height), random.choice(self.white_shades))
        draw = ImageDraw.Draw(img)
        draw.text((-bbox[0], -bbox[1]), text, font=font, fill=self.black)
        # Convert to OpenCV format and resize to standard height
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        aspect_ratio = cv_img.shape[1] / cv_img.shape[0]
        new_width = int(self.standard_height * aspect_ratio)
        return cv2.resize(cv_img, (new_width, self.standard_height))

    def adjust_brightness_contrast(self, image):
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.8, 1.2))
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.8, 1.2))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def rotate_image(self, image, angle):
        center = (image.shape[1]//2, image.shape[0]//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        new_width = int(image.shape[0] * abs_sin + image.shape[1] * abs_cos)
        new_height = int(image.shape[0] * abs_cos + image.shape[1] * abs_sin)
        rotation_matrix[0, 2] += (new_width // 2 - center[0])
        rotation_matrix[1, 2] += (new_height // 2 - center[1])
        return cv2.warpAffine(image, rotation_matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=random.choice(self.white_shades))

    def apply_zoom(self, image):
        zoom_factor = random.uniform(self.min_zoom, self.max_zoom)
        height, width = image.shape[:2]
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
        zoomed = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        if zoom_factor < 1:
            top_pad = (height - new_height) // 2
            left_pad = (width - new_width) // 2
            zoomed = cv2.copyMakeBorder(zoomed, top_pad, height - new_height - top_pad, left_pad, width - new_width - left_pad, cv2.BORDER_CONSTANT, value=random.choice(self.white_shades))
        else:
            start_y, start_x = (new_height - height) // 2, (new_width - width) // 2
            zoomed = zoomed[start_y:start_y + height, start_x:start_x + width]
        return zoomed

    def shear_image(self, image):
        shear_factor = random.uniform(-0.15, 0.15)
        height, width = image.shape[:2]
        new_width = width + int(abs(shear_factor * height))
        pts1 = np.float32([[0, 0], [width, 0], [0, height]])
        pts2 = np.float32([[int(shear_factor * height), 0], [width + int(shear_factor * height), 0], [0, height]])
        shear_matrix = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(image, shear_matrix, (new_width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=random.choice(self.white_shades))

    def apply_perspective_transform(self, image):
        height, width = image.shape[:2]
        margin = min(width, height) * 0.2
        src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        dst_points = np.float32([[random.uniform(0, margin), random.uniform(0, margin)],
                                 [random.uniform(width - margin, width), random.uniform(0, margin)],
                                 [random.uniform(0, margin), random.uniform(height - margin, height)],
                                 [random.uniform(width - margin, width), random.uniform(height - margin, height)]])
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=random.choice(self.white_shades))

    def save_label(self, filename, label):
        # Open the CSV file in append mode and write the label
        with open(self.labels_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([filename, label])  # Write the filename and label

    def process_word_file(self, input_file: str, augmentations_per_word: int = 15):
        with open(input_file, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()

        for i, word in tqdm(enumerate(words), desc="Processing words", total=len(words)):
            for j in range(augmentations_per_word):
                base_image = self.text_to_image(word)
                augmented = base_image.copy()
                if j < 7:
                    if random.random() < 0.5: augmented = self.rotate_image(augmented, random.uniform(-2.5, 2.5))
                    if random.random() < 0.6: augmented = self.adjust_brightness_contrast(augmented)
                    if random.random() < 0.4: augmented = self.apply_zoom(augmented)
                    if random.random() < 0.3: augmented = self.shear_image(augmented)
                elif j < 15:
                    if random.random() < 0.5: augmented = self.rotate_image(augmented, random.uniform(-3, 3))
                    if random.random() < 0.6: augmented = self.shear_image(augmented)
                    if random.random() < 0.6: augmented = self.apply_zoom(augmented)
                    if random.random() < 0.6: augmented = self.apply_perspective_transform(augmented)

                if augmented.shape[:2] != base_image.shape[:2]:
                    augmented = cv2.resize(augmented, (base_image.shape[1], base_image.shape[0]))
                
                filename = f'word_{i}_aug_{j}.png'
                output_path = os.path.join(self.output_dir, filename)
                cv2.imwrite(output_path, augmented)
                self.save_label(filename, word)  # Save the label

            # Generate 5 additional augmentations by combining the word with a second random word
            for k in range(5):
                second_word = random.choice(words)
                combined_image = self.combine_words(word, second_word)
                combined_label = f"{word} {second_word}"
                filename = f'word_{i}_aug_{k+15}.png'
                output_path_combined = os.path.join(self.output_dir, filename)
                cv2.imwrite(output_path_combined, combined_image)
                self.save_label(filename, combined_label) 

    def process_word_file(self, input_file: str, augmentations_per_word: int = 15):
        with open(input_file, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()

        for i, word in tqdm(enumerate(words), desc="Processing words", total=len(words)):
            for j in range(augmentations_per_word):
                base_image = self.text_to_image(word)
                augmented = base_image.copy()
                if j < 7:
                    if random.random() < 0.5: augmented = self.rotate_image(augmented, random.uniform(-2.5, 2.5))
                    if random.random() < 0.6: augmented = self.adjust_brightness_contrast(augmented)
                    if random.random() < 0.4: augmented = self.apply_zoom(augmented)
                    if random.random() < 0.3: augmented = self.shear_image(augmented)
                elif j < 15:
                    if random.random() < 0.5: augmented = self.rotate_image(augmented, random.uniform(-3, 3))
                    if random.random() < 0.6: augmented = self.shear_image(augmented)
                    if random.random() < 0.6: augmented = self.apply_zoom(augmented)
                    if random.random() < 0.6: augmented = self.apply_perspective_transform(augmented)

                if augmented.shape[:2] != base_image.shape[:2]:
                    augmented = cv2.resize(augmented, (base_image.shape[1], base_image.shape[0]))
                
                filename = f'word_{i}_aug_{j}.png'
                output_path = os.path.join(self.output_dir, filename)
                cv2.imwrite(output_path, augmented)
                self.save_label(filename, word)  # Save the label for the single word

            # Generate 5 additional augmentations by combining the word with a second random word
            for k in range(5):
                second_word = random.choice(words)
                combined_image = self.combine_words(word, second_word)
                combined_label = f"{word} {second_word}"

                filename = f'word_{i}_aug_{k+15}.png'
                output_path_combined = os.path.join(self.output_dir, filename)
                cv2.imwrite(output_path_combined, combined_image)
                
                # Save combined label correctly
                self.save_label(filename, combined_label)  # Save the combined label (word1 + word2)

    def combine_words(self, word1, word2):
        # Generate both images at standard height
        first_image = self.text_to_image(word1)
        second_image = self.text_to_image(word2)
        
        # Apply the same augmentations to both images
        augmented_first_image = first_image.copy()
        augmented_second_image = second_image.copy()

        # Apply the same augmentations pipeline for both images
        if random.random() < 0.5: 
            angle = random.uniform(-2.5, 2.5)
            augmented_first_image = self.rotate_image(augmented_first_image, angle)
            augmented_second_image = self.rotate_image(augmented_second_image, angle)

        if random.random() < 0.6:
            augmented_first_image = self.adjust_brightness_contrast(augmented_first_image)
            augmented_second_image = self.adjust_brightness_contrast(augmented_second_image)

        if random.random() < 0.4:
            augmented_first_image = self.apply_zoom(augmented_first_image)
            augmented_second_image = self.apply_zoom(augmented_second_image)

        if random.random() < 0.3:
            augmented_first_image = self.shear_image(augmented_first_image)
            augmented_second_image = self.shear_image(augmented_second_image)

        if random.random() < 0.6:
            augmented_first_image = self.apply_perspective_transform(augmented_first_image)
            augmented_second_image = self.apply_perspective_transform(augmented_second_image)

        # Resize both images to the standard height
        if augmented_first_image.shape[0] != self.standard_height:
            augmented_first_image = cv2.resize(augmented_first_image, 
                                            (augmented_first_image.shape[1], self.standard_height))
        if augmented_second_image.shape[0] != self.standard_height:
            augmented_second_image = cv2.resize(augmented_second_image, 
                                                (augmented_second_image.shape[1], self.standard_height))

        # Calculate space width (approximately 25% of average character width)
        avg_char_width = (augmented_first_image.shape[1] / len(word1) + augmented_second_image.shape[1] / len(word2)) / 2
        space_width = int(avg_char_width * 0.5)
        
        # Combine images horizontally with space
        combined_width = augmented_first_image.shape[1] + space_width + augmented_second_image.shape[1]
        combined_image = np.full((self.standard_height, combined_width, 3), fill_value=255, dtype=np.uint8)
        
        # Place the first and second images into the combined image
        combined_image[:, :augmented_first_image.shape[1]] = augmented_first_image
        combined_image[:, augmented_first_image.shape[1] + space_width:] = augmented_second_image
        
        return combined_image


def main():
    input_file = "kaithi_20k.txt"
    generator = ImageGenerator(output_dir='all_data')
    generator.process_word_file(input_file, augmentations_per_word=15)

if __name__ == "__main__":
    main()
