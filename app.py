# app.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import os

# CNN Model
class CNNModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 112 * 112, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Class names
class_names = ["Healthy", "Powdery", "Rust"]

# Image transform
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model based on user choice
def load_model(model_name):
    num_classes = 3
    if model_name == "CNN":
        model = CNNModel(num_classes)
        weights = "BestModel_CNN.pth"
    elif model_name == "ResNet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        weights = "BestModel_ResNet18.pth"
    elif model_name == "VGG16":
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        weights = "BestModel_VGG16.pth"
    elif model_name == "EfficientNet":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        weights = "BestModel_EfficientNet.pth"
    else:
        raise ValueError("Unsupported model")

    if not os.path.exists(weights):
        raise FileNotFoundError(f"Model weights not found: {weights}")

    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model

# Inference function
def predict(image, model_choice):
    try:
        model = load_model(model_choice)
        img = inference_transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img)
            probs = F.softmax(outputs[0], dim=0)
        return {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Leaf Image"),
        gr.Dropdown(["CNN", "ResNet18", "VGG16", "EfficientNet"], label="Choose Model")
    ],
    outputs=gr.Label(num_top_classes=3),
    title="Plant Disease Classifier",
    description="Upload a leaf image and choose a model to classify it as Healthy, Powdery, or Rust."
)

interface.launch()
