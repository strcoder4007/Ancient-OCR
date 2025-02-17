import cv2
import numpy as np
import os
from pathlib import Path

def deskew(thresh, max_skew=10):
    inverted = 255 - thresh
    coords = np.column_stack(np.where(inverted > 0))
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle += 90

    if abs(angle) > max_skew:
        return thresh

    (h, w) = thresh.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image(image_path, do_deskew=True):
    original = cv2.imread(image_path)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(bilateral, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 
                                   11, 2)
    if do_deskew:
        thresh = deskew(thresh)

    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return original, cleaned

def merge_vertical_lines_custom(vertical_mask, gap_threshold=10):
    contours, _ = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda b: b[0])
    
    merged_groups = []
    if not boxes:
        return vertical_mask.copy()
    current_group = [boxes[0]]
    for box in boxes[1:]:
        prev_box = current_group[-1]
        gap = box[0] - (prev_box[0] + prev_box[2])
        if gap < gap_threshold:
            current_group.append(box)
        else:
            merged_groups.append(current_group)
            current_group = [box]
    merged_groups.append(current_group)

    merged_mask = np.zeros_like(vertical_mask)
    for group in merged_groups:
        x_left = min(b[0] for b in group)
        x_right = max(b[0] + b[2] for b in group)
        x_center = (x_left + x_right) // 2
        y_top = min(b[1] for b in group)
        y_bottom = max(b[1] + b[3] for b in group)
        cv2.line(merged_mask, (x_center, y_top), (x_center, y_bottom), 255, 1)
    return merged_mask

def merge_horizontal_lines_custom(horizontal_mask, gap_threshold=10):
    contours, _ = cv2.findContours(horizontal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda b: b[1])  # sort by y coordinate (top to bottom)
    
    merged_groups = []
    if not boxes:
        return horizontal_mask.copy()
    current_group = [boxes[0]]
    for box in boxes[1:]:
        prev_box = current_group[-1]
        # Calculate vertical gap between current box and the bottom edge of previous box.
        gap = box[1] - (prev_box[1] + prev_box[3])
        if gap < gap_threshold:
            current_group.append(box)
        else:
            merged_groups.append(current_group)
            current_group = [box]
    merged_groups.append(current_group)
    
    # Create new mask and draw a 1-pixel high horizontal line at the center for each group.
    merged_mask = np.zeros_like(horizontal_mask)
    for group in merged_groups:
        y_top = min(b[1] for b in group)
        y_bottom = max(b[1] + b[3] for b in group)
        y_center = (y_top + y_bottom) // 2
        # Also, get the horizontal span for the group.
        x_left = min(b[0] for b in group)
        x_right = max(b[0] + b[2] for b in group)
        cv2.line(merged_mask, (x_left, y_center), (x_right, y_center), 255, 1)
    return merged_mask

def extract_vertical_lines(img_bin):
    h, w = img_bin.shape
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 60, 10)))
    temp = cv2.erode(img_bin, vertical_kernel, iterations=1)
    vertical = cv2.dilate(temp, vertical_kernel, iterations=2)

    full_vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical = cv2.dilate(vertical, full_vertical_kernel, iterations=1)

    vertical = merge_vertical_lines_custom(vertical, gap_threshold=2)
    return vertical

def extract_horizontal_lines(img_bin):
    h, w = img_bin.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 40, 5), 1))
    temp = cv2.erode(img_bin, horizontal_kernel, iterations=1)
    horizontal = cv2.dilate(temp, horizontal_kernel, iterations=2)

    full_horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, 1))
    horizontal = cv2.dilate(horizontal, full_horizontal_kernel, iterations=1)

    horizontal = merge_horizontal_lines_custom(horizontal, gap_threshold=10)
    return horizontal

def merge_lines(vertical, horizontal):
    """
    Combine vertical and horizontal line masks into a single table mask.
    """
    merged = cv2.bitwise_or(vertical, horizontal)
    # Additional close to fuse small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=1)
    return merged

def find_table_cells(original_img, table_mask):
    """
    Detect table cells via contours on the table_mask.
    Return bounding boxes (x, y, w, h).
    """
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    H, W, _ = original_img.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Basic size filtering
        if w < 20 or h < 20: 
            continue
        if w > 0.95 * W or h > 0.95 * H:
            continue

        # Check overlap with existing cells to avoid duplicates
        rect_area = w * h
        overlap_found = False
        for (ex, ey, ew, eh) in cells:
            overlap_w = max(0, min(x + w, ex + ew) - max(x, ex))
            overlap_h = max(0, min(y + h, ey + eh) - max(y, ey))
            overlap_area = overlap_w * overlap_h
            if overlap_area > 0.4 * rect_area:
                overlap_found = True
                break

        if not overlap_found:
            cells.append((x, y, w, h))

    # Sort cells top-to-bottom, then left-to-right
    cells.sort(key=lambda box: (box[1], box[0]))
    return cells

def extract_and_save_cells(image_path, output_folder="cropped_cells"):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    cells_folder = os.path.join(output_folder, "cells")
    Path(cells_folder).mkdir(exist_ok=True)

    # Step 1: Preprocess
    original, preprocessed = preprocess_image(image_path, do_deskew=True)
    cv2.imwrite(os.path.join(output_folder, "01_preprocessed.png"), preprocessed)

    # Step 2: Extract lines
    vertical = extract_vertical_lines(preprocessed)
    horizontal = extract_horizontal_lines(preprocessed)
    cv2.imwrite(os.path.join(output_folder, "02_vertical_lines.png"), vertical)
    cv2.imwrite(os.path.join(output_folder, "03_horizontal_lines.png"), horizontal)

    # Step 3: Merge lines
    table_mask = merge_lines(vertical, horizontal)
    cv2.imwrite(os.path.join(output_folder, "04_table_structure.png"), table_mask)

    # Step 4: Find table cells
    cells = find_table_cells(original, table_mask)

    # Step 5: Crop & save
    annotated = original.copy()
    for idx, (x, y, w, h) in enumerate(cells):
        cell_img = annotated[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(cells_folder, f"cell_{idx}.png"), cell_img)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_folder, "05_detected_cells.png"), annotated)
    print(f"Extracted {len(cells)} cells.")
    return len(cells)

if __name__ == "__main__":
    image_path = "test_images/real1.png"
    extract_and_save_cells(image_path, "cropped_cells")
