# 🚀 Install dependencies
!pip install -q gradio transformers torch torchvision

# 🧠 Imports
import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as T

# ⚠️ Load pickled model safely (entire model object)
model = torch.load("/content/ASL.pt", map_location="cpu", weights_only=False)
model.eval()

# 🔁 Manual preprocessing (resize, tensor, normalize)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),  # Converts to [C, H, W] in [0, 1]
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# 🔤 Class labels A-Z
labels = {str(i): chr(65 + i) for i in range(26)}

# 🔍 Inference function
def classify_sign(image):
    image = Image.fromarray(image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()

    top_idx = torch.argmax(probs).item()
    prediction = labels[str(top_idx)]
    confidence = round(probs[top_idx].item() * 100, 2)

    return f"{prediction} ({confidence}%)"

# 🎛️ Gradio Interface
interface = gr.Interface(
    fn=classify_sign,
    inputs=gr.Image(source="webcam", streaming=True, live=True),
    outputs=gr.Textbox(label="Detected Letter"),
    title="🧠 Real-Time Sign Language Translator",
    description="Live webcam-based ASL letter detection using a pre-trained model.",
    live=True
)

# 🚀 Launch the app
interface.launch(share=True)
