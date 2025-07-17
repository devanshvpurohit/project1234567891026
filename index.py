import gradio as gr
from PIL import Image
import torch
import torchvision.transforms as T

# Load the local PyTorch model
model_path = "/content/ASL.pt"
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()

# Manual preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),  # Converts to [C, H, W] in [0, 1]
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Class labels for A-Z (26 classes)
labels = {
    str(i): chr(65 + i) for i in range(26)
}

# Prediction function
def classify_sign(image):
    image = Image.fromarray(image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).squeeze()

    top_idx = torch.argmax(probs).item()
    prediction = labels[str(top_idx)]
    confidence = round(probs[top_idx].item() * 100, 2)

    return f"{prediction} ({confidence}%)"

# Gradio interface for webcam input
interface = gr.Interface(
    fn=classify_sign,
    inputs=gr.Image(source="webcam", streaming=True, live=True),
    outputs=gr.Textbox(label="Detected Letter"),
    title="🧠 Real-Time Sign Language Translator",
    description="Detects and classifies hand signs from the webcam in real time. Translates ASL letters into English text.",
    live=True
)

if __name__ == "__main__":
    interface.launch()
