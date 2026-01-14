"""
CIFAR-10 Image Classifier - Flask Web App
=========================================
Simple website to upload images and get predictions using a DALL-E inspired Transformer.

Usage:
    pip install flask
    python website.py
    Open: http://localhost:5000
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template_string, jsonify
import io
import base64
import os
import sys

# Handle truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# Model Definition
# ============================================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class DALLEInspiredClassifier(nn.Module):
    def __init__(self, img_size=32, num_classes=10):
        super().__init__()
        embed_dim, num_heads, num_layers = 384, 6, 6
        patch_size = 4
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))


# ============================================================
# Load Model
# ============================================================

print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate model with CIFAR-10 specs (32x32 input, 10 classes, patch_size=4)
model = DALLEInspiredClassifier(img_size=32, num_classes=10).to(device)

model_path = 'dalle_cifar10_best.pth'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
else:
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        # Check if state dict is nested
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path} on {device}")
    except Exception as e:
        print(f"Failed to load model weights: {e}")

model.eval()

# CIFAR-10 Labels
CIFAR10_CLASSES = [
    'Airplane ✈️', 'Automobile 🚗', 'Bird 🐦', 'Cat 🐱', 'Deer 🦌',
    'Dog 🐶', 'Frog 🐸', 'Horse 🐴', 'Ship 🚢', 'Truck 🚚'
]

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32 for this model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CIFAR-10 Image Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            color: #fff;
        }
        .container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 40px;
            max-width: 600px;
            width: 100%;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }
        h1 {
            background: linear-gradient(45deg, #ff00cc, #3333ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #aaa;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        .upload-area {
            border: 3px dashed rgba(100, 100, 255, 0.4);
            border-radius: 15px;
            padding: 40px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }
        .upload-area:hover {
            border-color: #3333ff;
            background: rgba(51, 51, 255, 0.1);
        }
        .upload-icon {
            font-size: 50px;
            margin-bottom: 15px;
        }
        .upload-text {
            color: #ccc;
            font-size: 1.1em;
        }
        #fileInput { display: none; }
        #preview {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            margin: 20px auto;
            display: none;
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        }
        .btn {
            background: linear-gradient(135deg, #3333ff, #ff00cc);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            border-radius: 30px;
            cursor: pointer;
            width: 100%;
            transition: all 0.2s;
            font-weight: bold;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(51, 51, 255, 0.4);
        }
        .btn:disabled {
            background: #444;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        #result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 15px;
            display: none;
            background: rgba(0, 0, 0, 0.2);
        }
        .result-label {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
            color: #fff;
        }
        .confidence {
            font-size: 1.1em;
            color: #ccc;
        }
        .loading {
            display: none;
            text-align: center;
            color: #3333ff;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid #3333ff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .arch-info {
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            padding: 10px;
            margin: 20px 0;
            font-size: 0.8em;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 CIFAR-10 Classifier</h1>
        <p class="subtitle">DALL-E Inspired Vision Transformer (From Scratch)</p>
        
        <div class="arch-info">
            <strong>Model Specs:</strong> 32x32 Input | 10 Classes | 11M Params
        </div>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">🖼️</div>
            <p class="upload-text">Click or Drop Image Here</p>
        </div>
        
        <input type="file" id="fileInput" accept="image/*">
        <img id="preview" alt="Preview">
        
        <button class="btn" id="predictBtn" disabled>Classify Image</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing with Neural Network...</p>
        </div>
        
        <div id="result">
            <div class="result-label" id="resultLabel"></div>
            <div class="confidence" id="confidence"></div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const predictBtn = document.getElementById('predictBtn');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');

        uploadArea.onclick = () => fileInput.click();
        
        uploadArea.ondragover = (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        };
        
        uploadArea.ondragleave = () => uploadArea.classList.remove('dragover');
        
        uploadArea.ondrop = (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFile(e.dataTransfer.files[0]);
            }
        };

        fileInput.onchange = (e) => {
            if (e.target.files.length) handleFile(e.target.files[0]);
        };

        function handleFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
                uploadArea.style.display = 'none';
                predictBtn.disabled = false;
                result.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        predictBtn.onclick = async () => {
            const file = fileInput.files[0];
            if (!file) return;
            
            loading.style.display = 'block';
            result.style.display = 'none';
            predictBtn.disabled = true;

            const formData = new FormData();
            formData.append('image', file);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                loading.style.display = 'none';
                result.style.display = 'block';
                
                if (data.error) {
                    alert(data.error);
                    return;
                }

                document.getElementById('resultLabel').textContent = data.params.predicted_class;
                document.getElementById('confidence').textContent = 
                    `Confidence: ${data.params.confidence.toFixed(1)}%`;
                    
            } catch (error) {
                loading.style.display = 'none';
                alert('Error: ' + error.message);
            }
            
            predictBtn.disabled = false;
        };
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    try:
        # Load image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Transform and predict
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
        
        pred_idx = outputs.argmax(dim=1).item()
        confidence = probs[pred_idx].item() * 100
        predicted_class = CIFAR10_CLASSES[pred_idx]
        
        # Get probs for all classes for debug/info
        class_probs = {CIFAR10_CLASSES[i]: float(probs[i]) for i in range(len(CIFAR10_CLASSES))}
        
        return jsonify({
            'params': {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'predictions': class_probs
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("CIFAR-10 Classifier Web App")
    print("="*50)
    print("\nOpen in browser: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
