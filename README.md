# Neural Steganography watermarking with Perceptual Losses

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

A research implementation of deep learning-based steganography using perceptual losses from VGG19/ResNet/MobileNetV2 to hide secret messages in images while preserving visual quality.

## Key Features
- 🖼️ Encoder-Decoder architecture for message embedding/extraction
- 🔍 Perceptual loss using pre-trained CNNs (VGG19/ResNet50/MobileNetV2)
- ⚡ Manual training loop for memory efficiency
- 📊 Quantitative metrics (PSNR, SSIM, BER)
- 🔐 Supports arbitrary length text messages

## Installation
```bash
git clone [https://github.com/Jireh-Jam/neural-steganography-watermarking.git]
cd neural-steganography
pip install -r requirements.txt

data/
├── train/
│   ├── img1.jpg
│   └── ...
└── val/
    ├── img1.jpg
    └── ...
from train import train_with_perceptual_loss

encoder, decoder = train_with_perceptual_loss(
    train_dir="data/train",
    val_dir="data/val",
    secret_message="MySecretKey123",
    epochs=50,
    batch_size=16
)
from utils import embed_message, extract_message

# Hide message
stego_image = embed_message(encoder, "image.jpg", "Hello World!")

# Recover message
secret = extract_message(decoder, "stego_image.png")

@software{Jireh-Jam_NeuralSteg,
  author = {Jireh Jam},
  title = {Neural Steganography with Perceptual Losses},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Jireh-Jam/watermarking}}
}
