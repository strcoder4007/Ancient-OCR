import os
import cv2
import numpy as np

def process_image(image_path, save_folder):
    img = cv2.imread(image_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    median_blurred = cv2.medianBlur(blurred, 5)
    
    bilateral_filtered = cv2.bilateralFilter(median_blurred, 9, 75, 75)
    
    _, thresh = cv2.threshold(bilateral_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    
    def draw_contours(image, contours):
        result_img = image.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.polylines(result_img, [np.array(contour)], True, (0, 255, 0), 1)
        return result_img

    result = draw_contours(img, contours)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    def crop_and_save(image, contours, save_folder, filename, padding=10):
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding

        x = max(x, 0)
        y = max(y, 0)
        h = min(h, image.shape[0] - y)
        w = min(w, image.shape[1] - x)

        cropped_img = image[y:y+h, x:x+w]

        crop_filename = os.path.join(save_folder, f"{filename}")
        cv2.imwrite(crop_filename, cropped_img)
        return cropped_img

    filename = os.path.basename(image_path)
    return crop_and_save(img, contours, save_folder, filename)

def process_images_in_folder(folder_path, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {image_name}")
            process_image(image_path, save_folder)


input_folder = 'extracted_images'
output_folder = 'cropped_images'

process_images_in_folder(input_folder, output_folder)
