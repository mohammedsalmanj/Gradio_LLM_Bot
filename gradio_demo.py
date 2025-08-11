import torch
from torchvision import transforms
import gradio as gr
import requests
from PIL import Image

# Load labels for ImageNet classes
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# Load pretrained ResNet18 model
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

def predict(inp):
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return confidences

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    examples=["lion.jpg", "cheetah.jpg"]  # Replace with local image paths if you want
)

iface.launch(server_name="127.0.0.1", server_port=7860)
