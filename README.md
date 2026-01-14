# � CIFAR-10 Image Classifier

A **from-scratch** implementation of a DALL-E-inspired Vision Transformer image classifier. This project uses a custom transformer architecture trained on the CIFAR-10 dataset to classify images into 10 categories.

![Model Type](https://img.shields.io/badge/Model-Vision%20Transformer-blueviolet)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-orange)
![Input Size](https://img.shields.io/badge/Input_Size-32x32-green)
![Patch Size](https://img.shields.io/badge/Patch_Size-4x4-yellow)

## 🎯 Features

- **Custom Vision Transformer**: Built from scratch using PyTorch.
- **CIFAR-10 Support**: Classifies images into 10 classes (Airplane, Car, Bird, Cat, deer, Dog, Frog, Horse, Ship, Truck).
- **Web Interface**: Modern Flask-based web UI for easy image testing.
- **CLI Tool**: Command-line script for quick predictions.

## 🚀 How to Run

### Prerequisites

```bash
pip install torch torchvision flask pillow
```

### 1. Run the Web App (Recommended)

This launches a local web server where you can upload images via a GUI.

```bash
python website.py
```

Then open **http://localhost:5000** in your browser.

### 2. Run Command Line Prediction

To classify a single image directly from your terminal:

```bash
python predict.py path/to/your/image.jpg
```

## 🏗️ Model Architecture

The model is a Vision Transformer (ViT) adapted for smaller images (CIFAR-10 resolution).

- **Input**: 32x32 pixels
- **Patches**: 4x4 pixels (Total 64 patches)
- **Embedding Dim**: 384
- **Heads**: 6
- **Layers**: 6
- **Parameters**: ~11 Million

## � Project Structure

```
cifar10-classifier/
├── website.py              # Flask Web Application
├── predict.py              # Command Line Prediction Tool
├── dalle_cifar10_best.pth  # Trained Model Weights
└── README.md
```

## 📄 License

MIT License
