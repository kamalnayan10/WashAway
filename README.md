# WashAway: AI-Powered Watermark Remover

WashAway is an advanced deep learning tool built to automatically remove watermarks from images. Utilizing the power of PyTorch and the UNET architecture from Pix2Pix, WashAway offers an efficient and precise solution for image restoration through a Generative AI model. The model has been trained on a custom-made, meticulously crafted dataset, ensuring exceptional performance across a variety of image types and watermark designs.

# How It Works

WashAway is based on the UNET architecture implemented with PyTorch and utilizes the Pix2Pix framework for image-to-image translation. Given a watermarked image, the model removes the watermark by learning to predict and generate the original image underneath.

# Custom Dataset

The dataset used for training was custom-built, comprising a variety of watermarked images and their corresponding clean versions. This ensures the model generalizes well across different watermark styles and image complexities.
