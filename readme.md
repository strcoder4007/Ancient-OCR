## How to run [New]
1. Start by running ```extract_images.py``` to extract images from excel file.
2. Then run ```crop_images.py``` to get cropped images using contours.
2. Then run ```process_images.py``` to clean the images and generate augmentations.
3. Run ```split.py``` to split data into train and validation set.



## How to run [OLD]
1. Start by running ```get_data.py```, this will get the data from hugging face wiki and save top 50,000 words in ```kaithi_50000.txt```
2. Then run ```process_images.py``` and ```generate_labels.py```
3. Run ```split.py``` to split data into train and validation set.