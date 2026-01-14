"""
CIFAR-10 Classifier - Inference Script
======================================
Use the trained model to predict on new images.

Usage:
    python predict.py path/to/image.jpg
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os

# Handle truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# Model Definition (same as website.py)
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
# Prediction Function
# ============================================================

def predict(image_path, model_path='dalle_cifar10_best.pth'):
    """Predict using the CIFAR-10 model."""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = DALLEInspiredClassifier(img_size=32, num_classes=10).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        sys.exit(1)
        
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
         # Check if state dict is nested
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and process image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    pred_idx = outputs.argmax(dim=1).item()
    confidence = probs[pred_idx].item() * 100
    
    classes = [
        'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
        'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
    ]
    
    return classes[pred_idx], confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    
    prediction, confidence = predict(image_path)
    
    print(f"\n{'='*50}")
    print(f"Image: {image_path}")
    print(f"Prediction: {prediction.upper()} 🔍")
    print(f"Confidence: {confidence:.1f}%")
    print(f"{'='*50}")
