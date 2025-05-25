# WashAway: AI-Powered Watermark Remover

**WashAway** is a deep learning-powered tool for removing watermarks from images, using a UNet-based generator and a discriminator architecture within a Pix2Pix framework. It features a simple and intuitive **Streamlit** web interface and is backed by pretrained models trained on a custom dataset.

---

## 🚀 Features

* 🧠 **Deep Learning Architecture**: UNet-based Generator and CNN Discriminator using Pix2Pix principles.
* 🖼️ **Image-to-Image Translation**: Removes visible watermarks with high fidelity.
* 🧪 **Streamlit UI**: Clean, interactive web interface for quick testing.
* 🗂️ **Pretrained Models**: Quickly demo the app with pretrained weights.

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kamalnayan10/WashAway.git
cd WashAway
```

### 2. Create and Activate a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Download Pretrained Weights

The generator and discriminator weights are available here:

📁 [Google Drive - Model Weights](https://drive.google.com/drive/folders/1rERvSJJeR-DjFMHycWT1_QjVXULk3xth?usp=sharing)

Download and place the weights in the appropriate directory (as expected by your code, typically something like `./checkpoints/`).

---

## 🧑‍💻 Usage

### Run the Web App

```bash
streamlit run app.py
```

### Input

* Upload a watermarked image using the app interface.
* The model will process the image and return a cleaned version without the watermark.

---

## 🧠 Model Architecture

* **Generator**: UNet-based architecture for image restoration.
* **Discriminator**: PatchGAN-style CNN to distinguish between real and generated images.

---

## 📂 Project Structure

```
WashAway/
├── app.py                # Streamlit app
├── washaway.py           # Inference script
├── gen.py                # Generator model
├── disc.py               # Discriminator model
├── train.py              # Training loop
├── dataset.py            # Dataset loader
├── utils.py              # Utility functions
├── config.py             # Configuration and hyperparameters
├── requirements.txt      # Python dependencies
└── checkpoints/          # Directory to place downloaded weights
```

---

## 🤝 Contributing

Feel free to fork the repo and submit a pull request for improvements or bug fixes!

1. Fork it
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙋‍♂️ Maintainers

Created by [Kamal Nayan](https://github.com/kamalnayan10)
Maintained with minor contributions by friends and the open-source community.

---
