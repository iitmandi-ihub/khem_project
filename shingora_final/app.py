from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import os

# Define the class labels
class_labels = ['Hairy_Thick_Thin', 'NAPS', 'PURE', 'SLUBS']

# Define the model class
class ResNet50Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Model, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Initialize Flask app
app = Flask(__name__)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50Model(num_classes=4)
model.load_state_dict(torch.load('best_resnet50_model_4_class.pth', map_location=device))
model.to(device)
model.eval()

# Define the transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the function to predict a single image
def predict_image(image_path, class_labels):
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_idx = predicted.item()
        predicted_label = class_labels[predicted_idx]
    
    return predicted_idx, predicted_label

# Create an API route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in the request'}), 400

    image_file = request.files['image']
    
    # Save the image to a temporary location
    image_path = './temp_image.jpg'
    image_file.save(image_path)

    try:

        # Get prediction for the image
        predicted_class_idx, predicted_label = predict_image(image_path, class_labels)

        response = {
            'success': True,
            'message': 'Prediction successful',
            'predicted_class': predicted_class_idx,
            'predicted_label': predicted_label
        }

        return jsonify(response), 200 
    except Exception as e:
        return jsonify({
            'success': False,
            'message':f'{str(e)}',
        }), 500
    finally:
        # Clean up temporary image
        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == '__main__':
    app.run(debug=True)