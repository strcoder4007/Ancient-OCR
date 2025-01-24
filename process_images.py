import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import random
from tqdm import tqdm
import csv

class ImageGenerator:
    def __init__(self, input_dir: str, labels_file: str, output_dir: str = 'all_data', min_zoom: float = 0.5, max_zoom: float = 1.0):
        self.input_dir = input_dir
        self.labels_file = labels_file
        self.output_dir = output_dir
        self.min_zoom, self.max_zoom = min_zoom, max_zoom
        os.makedirs(output_dir, exist_ok=True)
        self.white_shades = [(255, 255, 255), (250, 250, 250)]
        self.black = (0, 0, 0)
        self.output_labels_file = os.path.join(output_dir, 'labels.csv')
        
        with open(self.output_labels_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'words'])

        self.standard_height = 64
        self.load_input_labels()

    def load_input_labels(self):
        self.input_labels = {}
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.input_labels[row[0]] = row[1]

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
        
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        aspect_ratio = gray_img.shape[1] / gray_img.shape[0]
        new_width = int(self.standard_height * aspect_ratio)
        return cv2.resize(gray_img, (new_width, self.standard_height))

    def adjust_brightness_contrast(self, image):
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.8, 1.2))
        pil_img = self.remove_noise_using_contours(pil_img)
        grayscale_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        return grayscale_img

    def rotate_image(self, image, angle):
        center = (image.shape[1]//2, image.shape[0]//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        new_width = int(image.shape[0] * abs_sin + image.shape[1] * abs_cos)
        new_height = int(image.shape[0] * abs_cos + image.shape[1] * abs_sin)
        rotation_matrix[0, 2] += (new_width // 2 - center[0])
        rotation_matrix[1, 2] += (new_height // 2 - center[1])
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        rotated_image = self.remove_noise_using_contours(rotated_image)
        if len(rotated_image.shape) == 3:
            return cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        else:
            return rotated_image

    def remove_noise_using_contours(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        result = cv2.bitwise_and(image, image, mask=mask)
        return result

    def apply_zoom(self, image):
        zoom_factor = random.uniform(self.min_zoom, self.max_zoom)
        height, width = image.shape[:2]
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
        zoomed = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        if zoom_factor < 1:
            top_pad = (height - new_height) // 2
            left_pad = (width - new_width) // 2
            zoomed = cv2.copyMakeBorder(zoomed, top_pad, height - new_height - top_pad, left_pad, width - new_width - left_pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        else:
            start_y, start_x = (new_height - height) // 2, (new_width - width) // 2
            zoomed = zoomed[start_y:start_y + height, start_x:start_x + width]
        zoomed = self.remove_noise_using_contours(zoomed)
        if len(zoomed.shape) == 3:
            return cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
        else:
            return zoomed

    def shear_image(self, image):
        shear_factor = random.uniform(-0.15, 0.15)
        height, width = image.shape[:2]
        new_width = width + int(abs(shear_factor * height))
        pts1 = np.float32([[0, 0], [width, 0], [0, height]])
        pts2 = np.float32([[int(shear_factor * height), 0], [width + int(shear_factor * height), 0], [0, height]])
        shear_matrix = cv2.getAffineTransform(pts1, pts2)
        sheared_image = cv2.warpAffine(image, shear_matrix, (new_width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        sheared_image = self.remove_noise_using_contours(sheared_image)
        if len(sheared_image.shape) == 3:
            return cv2.cvtColor(sheared_image, cv2.COLOR_BGR2GRAY)
        else:
            return sheared_image

    def apply_perspective_transform(self, image):
        height, width = image.shape[:2]
        margin = min(width, height) * 0.2
        src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        dst_points = np.float32([[random.uniform(0, margin), random.uniform(0, margin)],
                                 [random.uniform(width - margin, width), random.uniform(0, margin)],
                                 [random.uniform(0, margin), random.uniform(height - margin, height)],
                                 [random.uniform(width - margin, width), random.uniform(height - margin, height)]])
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed_image = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        transformed_image = self.remove_noise_using_contours(transformed_image)
        if len(transformed_image.shape) == 3:
            return cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
        else:
            return transformed_image

    def save_label(self, filename, label):
        with open(self.output_labels_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([filename, label])

    def apply_low_resolution(self, image):
        height, width = image.shape[:2]
        scale_factor = random.choice([3, 6])        
        low_res = cv2.resize(image, (width // scale_factor, height // scale_factor), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(low_res, (width, height), interpolation=cv2.INTER_LINEAR)

    def apply_scanner_noise(self, image):
        """Apply realistic scanner noise to the image"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if random.random() < 0.8:
            noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
            
            if noise_type == 'gaussian':
                mean = 0
                var = random.uniform(0.001, 0.02)
                sigma = var ** 0.5
                gaussian = np.random.normal(mean, sigma, image.shape) * 255
                noisy = image.astype(float) + gaussian
                noisy = np.clip(noisy, 0, 255).astype(np.uint8)
                return noisy
                
            elif noise_type == 'salt_pepper':
                amount = random.uniform(0.001, 0.01)
                s_vs_p = 0.5
                out = image.copy()
                
                # Salt noise
                num_salt = np.ceil(amount * image.size * s_vs_p)
                salt_coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
                out[tuple(salt_coords)] = 255
                
                # Pepper noise
                num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
                pepper_coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
                out[tuple(pepper_coords)] = 0
                return out
                
            elif noise_type == 'speckle':
                speckle = np.random.randn(*image.shape)
                noisy = image.astype(float) + image.astype(float) * speckle * 0.1
                noisy = np.clip(noisy, 0, 255).astype(np.uint8)
                return noisy
        
        return image

    def process_images_in_folder(self, augmentations_per_image: int = 15):
        image_files = [f for f in os.listdir(self.input_dir) if f.endswith('.png')]

        for i, filename in tqdm(enumerate(image_files), desc="Processing images", total=len(image_files)):
            image_path = os.path.join(self.input_dir, filename)
            base_image = cv2.imread(image_path)
            
            if filename in self.input_labels:
                label = self.input_labels[filename]
            else:
                label = "Unknown"

            for j in range(augmentations_per_image):
                augmented = base_image.copy()
                
                if random.random() < 0.7: augmented = self.rotate_image(augmented, random.uniform(-2.5, 2.5))
                if random.random() < 0.7: augmented = self.apply_zoom(augmented)
                if random.random() < 0.7: augmented = self.shear_image(augmented)

                if j >= 7:
                    if random.random() < 0.7: augmented = self.rotate_image(augmented, random.uniform(-3.5, 3.5))
                    if random.random() < 0.6: augmented = self.shear_image(augmented)
                    if random.random() < 0.6: augmented = self.apply_zoom(augmented)
                    if random.random() < 0.6: augmented = self.apply_perspective_transform(augmented)

                # Apply scanner-style augmentations
                augmented = self.apply_scanner_noise(augmented)
                if random.random() < 0.8: augmented = self.apply_low_resolution(augmented)

                if augmented.shape[:2] != base_image.shape[:2]:
                    augmented = cv2.resize(augmented, (base_image.shape[1], base_image.shape[0]))

                output_filename = f'{filename.split(".")[0]}_aug_{j}.png'
                output_path = os.path.join(self.output_dir, output_filename)
                
                if len(augmented.shape) == 3:
                    augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)

                cv2.imwrite(output_path, augmented)
                self.save_label(output_filename, label)

def main():
    input_dir = "cropped_images"
    labels_file = "labels.csv"
    generator = ImageGenerator(input_dir=input_dir, labels_file=labels_file, output_dir='all_data')
    generator.process_images_in_folder(augmentations_per_image=15)

if __name__ == "__main__":
    main()