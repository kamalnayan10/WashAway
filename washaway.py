import torch
from config import *
from utils import *
from gen import Generator
import torch.optim as optim
from test_dataset import WatermarkDataset
from image_patching import ImageSegmentation
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import os


def delete_non_empty_folder(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    
    os.rmdir(folder_path)


def stitch_patches(img , SIZE = 512):

    num_x = (img.width + SIZE - 1) // SIZE  # Ceiling division
    num_y = (img.height + SIZE - 1) // SIZE  # Ceiling division

    stitched_image = Image.new('RGB', (img.width, img.height))

    for t in range(num_y):
        for l in range(num_x):
            patch = Image.open(f"test_input/patch_{t}_{l}.jpg")

            top = t * SIZE
            left = l * SIZE
            right = min(left + SIZE, img.width)
            bottom = min(top + SIZE, img.height)

            side_gap = (SIZE - (right - left))/2

            top_gap = (SIZE - (bottom - top))/2

            crop_box = (side_gap , top_gap, right - side_gap, bottom - top_gap)
            patch = patch.crop(crop_box)

            stitched_image.paste(patch, (left, top))

    stitched_image.save(f"test_output.jpg")
    print("Image stitched back!")


def remove_watermark(gen , img_path):

    if not os.path.exists("test_input"):
        os.makedirs("test_input")

    image = ImageSegmentation(img_path , dir = "test_input" , out_img_path="test")
    image.create_patches()

    img_dataset = WatermarkDataset("test_input")
    img_loader = DataLoader(img_dataset, batch_size=1)

    gen.eval()

    with torch.no_grad():
        for i, (input_image , img_file) in enumerate(img_loader):
            out_img = gen(input_image.to(DEVICE))
            out_img = out_img * 0.5 + 0.5 #denormalising
            name = str(img_file)[2:-3]

            save_image(out_img, "test_input" + f"/{name}")

    stitch_patches(Image.open(img_path) , SIZE = image.SIZE)

    delete_non_empty_folder("test_input")


if __name__ == "__main__":
    gen = Generator().to(DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE)

    remove_watermark(gen , "images/watermarked.jpg")
