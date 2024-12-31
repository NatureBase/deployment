from flask import Flask, request, jsonify, render_template
from transformers import AutoImageProcessor
from PIL import Image
import torch
import torch.nn as nn
from transformers import ViTModel

# Define SpeciesSpecificAttention class
class SpeciesSpecificAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SpeciesSpecificAttention, self).__init__()
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask = self.sigmoid(self.fc(x))  # Generate attention mask
        return mask

# Define ViTWithSSA class
class ViTWithSSA(nn.Module):
    def __init__(self, num_classes):
        super(ViTWithSSA, self).__init__()

        # Load Hugging Face Vision Transformer (ViT) Model
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")  

        # Get embedding dimension dynamically
        embed_dim = self.vit.config.hidden_size

        # Add Species-Specific Attention (SSA)
        self.ssa = SpeciesSpecificAttention(embed_dim=embed_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # Pass input through ViT model
        outputs = self.vit(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state

        # Separate class token and patch embeddings
        cls_token, patches = hidden_states[:, 0], hidden_states[:, 1:]

        # Apply SSA to patch embeddings
        mask = self.ssa(patches)  # Compute attention mask
        patches = patches * mask  # Apply mask

        # Combine class token and modulated patches
        x = torch.cat([cls_token.unsqueeze(1), patches], dim=1)

        # Classification head
        x = x[:, 0]  # Extract class token
        x = self.classifier(x)  # Final classification
        return x

# Load model and processor
model = ViTWithSSA(num_classes=9)  # Replace with your actual class count
checkpoint_path = "/path/to/checkpoint.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")['model_state_dict'])
model.eval()

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
label_mapping = {0: "Sea Bass", 1: "Trout", 2: "Horse Mackerel"}  # Add your actual mappings

app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    try:
        # Process image
        image = Image.open(file).convert("RGB")
        pixel_values = image_processor(image, return_tensors="pt")['pixel_values']
        with torch.no_grad():
            outputs = model(pixel_values)
            _, predicted_class = torch.max(outputs, dim=1)
        label = label_mapping[predicted_class.item()]
        return jsonify({'prediction': label}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
