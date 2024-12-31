from flask import Flask, request, jsonify, render_template
from transformers import AutoImageProcessor
from PIL import Image
import torch
import torch.nn as nn
from transformers import ViTModel
import requests
import os
import torch.serialization
import pickle
import gdown


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

torch.serialization.add_safe_globals({
    'OrderedDict': torch.nn.Module
})

# Download checkpoint from Google Drive if not exists
# https://drive.google.com/file/d/1OgxWPmw4mvbmr76Y4OA1kwwP4Nt4rbyo/view?usp=sharing
# checkpoint_url = "https://drive.google.com/uc?id=1OgxWPmw4mvbmr76Y4OA1kwwP4Nt4rbyo&export=download"
# checkpoint_path = "checkpoint.pth"

file_id = "1OgxWPmw4mvbmr76Y4OA1kwwP4Nt4rbyo"
output_path = "checkpoint.pth"
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Now you can load the model state
model = ViTWithSSA(num_classes=9)  # Replace with your actual class count

# Define a custom pickle module with proper handling for encoding
class CustomPickleModule:
    @staticmethod
    def load(file, **kwargs):
        # Use the custom Unpickler explicitly to handle the encoding
        return CustomPickleModule.Unpickler(file, **kwargs).load()

    class Unpickler(pickle.Unpickler):
        def __init__(self, file, *args, **kwargs):
            kwargs.pop("encoding", None)  # Remove unsupported encoding argument
            super().__init__(file, *args, **kwargs)

        def find_class(self, module, name):
            # Optionally customize or restrict class loading here
            return super().find_class(module, name)

# Load the checkpoint using the custom pickle module
with open("checkpoint.pth", "rb") as f:
    checkpoint = torch.load(f, map_location="cpu", encoding='latin1', pickle_module=CustomPickleModule)


# model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")['model_state_dict'])
# checkpoint = torch.load(destination, map_location="cpu", pickle_module=pickle)
model.load_state_dict(checkpoint)
model.eval()



image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
label_mapping = label_mapping = {
    0: "Sea Bass",
    1: "Trout",
    2: "Horse Mackerel",
    3: "Shrimp",
    4: "Gilt-Head Bream",
    5: "Red Sea Bream",
    6: "Striped Red Mullet",
    7: "Red Mullet",
    8: "Black Sea Sprat"
}  # Add your actual mappings

app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    try:
        # Process the file
        image = Image.open(file).convert("RGB")
        pixel_values = image_processor(image, return_tensors="pt")['pixel_values']
        with torch.no_grad():
            outputs = model(pixel_values)
            _, predicted_class = torch.max(outputs, dim=1)
            print(f"Predicted class index: {predicted_class.item()}")

            # Ensure valid mapping
            label = label_mapping.get(predicted_class.item(), "Unknown Class")
            print(f"Predicted label: {label}")

        return jsonify({'prediction': label}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)