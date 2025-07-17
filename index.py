# 📦 Install dependencies
pip install -q gradio onnxruntime numpy opencv-python gTTS

# ✅ Import modules
import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image
from gtts import gTTS
import os

# ✅ Load the ONNX model
MODEL_PATH = "/content/ASL.onnx"  # Make sure this file exists in your Colab
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ✅ ASL Class Labels (A–Z)
LABELS = [chr(i) for i in range(65, 91)]  # A-Z

# ✅ Preprocess image
def preprocess(frame):
    image = Image.fromarray(frame).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# ✅ Inference + TTS
def recognize_and_speak(frame, speak):
    input_tensor = preprocess(frame)
    outputs = session.run([output_name], {input_name: input_tensor})[0]
    
    pred_idx = np.argmax(outputs, axis=1)[0]
    confidence = float(np.max(outputs))
    label = LABELS[pred_idx]
    result_text = f"Prediction: {label} ({round(confidence * 100, 2)}%)"

    audio_path = None
    if speak:
        tts = gTTS(text=label, lang='en')
        audio_path = "/content/tts_output.mp3"
        tts.save(audio_path)

    return result_text, (audio_path if speak else None)

# ✅ Gradio App
with gr.Blocks(title="ASL Gesture Recognizer (ONNX)") as demo:
    gr.Markdown("# 🤟 ASL Gesture Recognizer")
    gr.Markdown("Live ASL letter prediction using webcam and ONNX model in Colab.")

    with gr.Row():
        webcam = gr.Image(label="📷 Live Webcam", source="webcam", tool=None)
        speak_checkbox = gr.Checkbox(label="🔊 Enable Text-to-Speech")

    output_text = gr.Textbox(label="✍️ Recognized Letter")
    audio_output = gr.Audio(label="🔈 Audio Output", type="file")

    webcam.change(
        recognize_and_speak,
        inputs=[webcam, speak_checkbox],
        outputs=[output_text, audio_output],
        every=1.5  # Run every 1.5 seconds
    )

demo.launch(share=True)
