import streamlit as st
import torch
from config import *
from utils import load_checkpoint
from gen import Generator
import torch.optim as optim
from test_dataset import WatermarkDataset
from image_patching import ImageSegmentation
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import os
import shutil

# Helper functions
def delete_non_empty_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def stitch_patches(img, SIZE=512):
    num_x = (img.width + SIZE - 1) // SIZE
    num_y = (img.height + SIZE - 1) // SIZE
    stitched_image = Image.new('RGB', (img.width, img.height))

    for t in range(num_y):
        for l in range(num_x):
            patch = Image.open(f"test_input/patch_{t}_{l}.jpg")
            top = t * SIZE
            left = l * SIZE
            right = min(left + SIZE, img.width)
            bottom = min(top + SIZE, img.height)

            side_gap = (SIZE - (right - left)) / 2
            top_gap = (SIZE - (bottom - top)) / 2

            crop_box = (side_gap, top_gap, right - side_gap, bottom - top_gap)
            patch = patch.crop(crop_box)

            stitched_image.paste(patch, (left, top))

    out_path = "test_output.jpg"
    stitched_image.save(out_path)
    return out_path

def remove_watermark(gen, img_path):
    if not os.path.exists("test_input"):
        os.makedirs("test_input")

    image = ImageSegmentation(img_path, dir="test_input", out_img_path="test")
    image.create_patches()

    img_dataset = WatermarkDataset("test_input")
    img_loader = DataLoader(img_dataset, batch_size=1)

    gen.eval()
    with torch.no_grad():
        for i, (input_image, img_file) in enumerate(img_loader):
            out_img = gen(input_image.to(DEVICE))
            out_img = out_img * 0.5 + 0.5  # denormalizing
            name = str(img_file)[2:-3]
            save_image(out_img, f"test_input/{name}")

    out_path = stitch_patches(Image.open(img_path), SIZE=image.SIZE)
    delete_non_empty_folder("test_input")
    return out_path

# Streamlit App
st.title("🧼 Watermark Remover using GAN")

uploaded_file = st.file_uploader("Upload a watermarked image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with open("temp_input.jpg", "wb") as f:
        f.write(uploaded_file.read())

    st.image("temp_input.jpg", caption="Uploaded Image", use_column_width=True)

    if st.button("Remove Watermark"):
        st.write("Processing... Please wait.")
        gen = Generator().to(DEVICE)
        opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE)

        output_path = remove_watermark(gen, "temp_input.jpg")
        st.image(output_path, caption="Result", use_column_width=True)
        with open(output_path, "rb") as file:
            st.download_button(label="📥 Download Image", data=file, file_name="output.jpg", mime="image/jpeg")

        os.remove("temp_input.jpg")
        os.remove(output_path)
