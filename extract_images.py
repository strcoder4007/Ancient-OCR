import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl_image_loader import SheetImageLoader

wb = load_workbook('KaithiDataset.xlsx')
ws = wb.active

image_path = 'extracted_images'

os.makedirs(image_path, exist_ok=True)
labels_file = 'labels.csv'

labels_data = []

image_loader = SheetImageLoader(ws)

for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
    label = row[0] 
    for col_idx in range(2, 5): 
        if image_loader.image_in(f'{chr(64 + col_idx)}{row_idx}'): 
            img = image_loader.get(f'{chr(64 + col_idx)}{row_idx}') 
            img_name = f'img{len(labels_data) + 1}.png'
            img_path = os.path.join(image_path, img_name)
            
            img.save(img_path)
            labels_data.append((img_name, label))

labels_df = pd.DataFrame(labels_data, columns=['image', 'label'])
labels_df.to_csv(labels_file, index=False)

print(f"Images and labels have been saved. Labels CSV: {labels_file}")