"""
This is a temporary python script I made to populate training data
"""

from PIL import Image
import os

# Paths to the folders
fold1_path = 'patches_nonwatermarked'
fold2_path = 'patches_watermarked'
output_path = 'train'

COUNT = 154

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Get the list of images in both folders
images_fold1 = sorted(os.listdir(fold1_path))
images_fold2 = sorted(os.listdir(fold2_path))

# Ensure both folders have the same number of images
if len(images_fold1) != len(images_fold2):
    raise ValueError("The folders do not contain the same number of images.")


# Iterate through the images and concatenate them
for img_name1, img_name2 in zip(images_fold1, images_fold2):
    img1 = Image.open(os.path.join(fold1_path, img_name1))
    img2 = Image.open(os.path.join(fold2_path, img_name2))

    # Ensure both images have the same size
    if img1.size != img2.size:
        raise ValueError(f"Image sizes do not match for {img_name1} and {img_name2}.")

    # Concatenate images side by side (horizontally)
    concatenated_image = Image.new('RGB', (img1.width + img2.width, img1.height))
    concatenated_image.paste(img1, (0, 0))
    concatenated_image.paste(img2, (img1.width, 0))

    # Save the concatenated image
    concatenated_image.save(os.path.join(output_path, f"img_{COUNT}.jpg"))

    COUNT += 1

print("Image concatenation completed successfully!")