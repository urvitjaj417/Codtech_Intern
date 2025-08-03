import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import os

# 🧠 Load and preprocess image
def load_image(path, max_size=400, shape=None):
    image = Image.open(path).convert("RGB")
    
    if shape is not None:
        if isinstance(shape, torch.Size):
            shape = tuple(shape)
        resize_shape = shape
    else:
        size = max(image.size)
        resize_shape = (size, size)

    transform = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image

# 📐 Compute gram matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# 🔍 Extract CNN features
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content layer
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# 🖌️ Neural Style Transfer function
def style_transfer(content_path, style_path, steps=300, style_weight=1e6, content_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = models.vgg19(pretrained=True).features.to(device).eval()

    content = load_image(content_path).to(device)
    style = load_image(style_path, shape=content.shape[-2:]).to(device)
    target = content.clone().requires_grad_(True).to(device)

    style_features = get_features(style, vgg)
    content_features = get_features(content, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    optimizer = optim.Adam([target], lr=0.003)

    for i in range(steps):
        optimizer.zero_grad()
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        style_loss = 0
        for layer in style_features:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            layer_loss = torch.mean((target_gram - style_gram)**2)
            b, c, h, w = target_features[layer].shape
            style_loss += layer_loss / (c * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        optimizer.step()

    final = target.cpu().clone().squeeze()
    final = final.detach().numpy().transpose(1, 2, 0)
    final = final * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    final = final.clip(0, 1)
    return final

# 🖼️ GUI app class
class StyleTransferApp:
    def __init__(self):
        self.content_path = None
        self.style_path = None
        self.root = tk.Tk()
        self.root.title("Neural Style Transfer")
        self.root.geometry("400x200")

        tk.Button(self.root, text="Select Content Image", command=self.load_content).pack(pady=5)
        tk.Button(self.root, text="Select Style Image", command=self.load_style).pack(pady=5)
        tk.Button(self.root, text="Start Transfer", command=self.run_transfer).pack(pady=10)

        self.status = tk.Label(self.root, text="Ready")
        self.status.pack()

        self.root.mainloop()

    def load_content(self):
        self.content_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if self.content_path:
            self.status.config(text=f"Content Loaded: {os.path.basename(self.content_path)}")

    def load_style(self):
        self.style_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if self.style_path:
            self.status.config(text=f"Style Loaded: {os.path.basename(self.style_path)}")

    def run_transfer(self):
        if not self.content_path or not self.style_path:
            messagebox.showwarning("Missing Input", "Please select both content and style images.")
            return
        self.status.config(text="Transferring Style...")
        self.root.update()

        try:
            # Check if files exist before running transfer
            if not os.path.isfile(self.content_path):
                messagebox.showerror("Error", f"Content image not found:\n{self.content_path}")
                self.status.config(text="Failed")
                return
            if not os.path.isfile(self.style_path):
                messagebox.showerror("Error", f"Style image not found:\n{self.style_path}")
                self.status.config(text="Failed")
                return

            result = style_transfer(self.content_path, self.style_path)
            plt.figure()
            plt.imshow(result)
            plt.title("Styled Output")
            plt.axis('off')
            plt.show()
            self.status.config(text="Done!")
        except Exception as e:
            messagebox.showerror("Error", f"Style transfer failed:\n{str(e)}")
            self.status.config(text="Failed")

# 🚀 Launch the app
StyleTransferApp()