import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms, models
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import gdown

def download_model_if_needed():
    model_path = "models/efficientnet_cardamom_best.pth"
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        file_id = "1Byc95R6YMbAcrYIQ3sGLV04OLwBlA_CJ"
        url = f"https://drive.google.com/uc?id={file_id}"
        print("🔽 Downloading model...")
        gdown.download(url, model_path, quiet=False)
        print("✅ Model downloaded.")


# --------------------------
# Config
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --------------------------
# Load Model
# --------------------------
@st.cache_resource
def load_model(num_classes):
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("models/efficientnet_cardamom_best.pth", map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

# --------------------------
# Prediction Function
# --------------------------
def predict_image(image, model):
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output[0], dim=0).cpu().numpy()
    return probs

# --------------------------
# Sensor Label Logic
# --------------------------
def get_sensor_label(weight, temp, rh):
    if weight >= 0.75 and rh >= 50 and temp <= 40:
        return "raw"
    elif weight >= 0.5 and rh >= 35 and 45 < temp <= 50:
        return "partially_dried"
    elif weight >= 0.3 and rh >= 25 and 50 < temp <= 60:
        return "mostly_dried"
    else:
        return "fully_dried"

# --------------------------
# Heat Pump Control Logic
# --------------------------
def get_control_signal(current, previous):
    stages = ["raw", "partially_dried", "mostly_dried", "fully_dried"]
    if not previous:
        return "START"
    diff = stages.index(current) - stages.index(previous)
    if diff > 0:
        return "MAINTAIN"
    elif diff == 0:
        return "STABLE"
    else:
        return "REDUCE_TEMP"

def main():
    download_model_if_needed()

    st.set_page_config(page_title="Cardamom Drying Simulation", layout="centered")
    st.title("🌿 Cardamom Drying Simulation with Heat Pump Control")

    if "logs" not in st.session_state:
        st.session_state.logs = []
        st.session_state.last_stage = None

    st.subheader("Step 1: Image & Sensor Input")
    image = st.file_uploader("📤 Upload image", type=["jpg", "jpeg", "png"])
    weight = st.number_input("Weight (kg)", min_value=0.0, max_value=1.0, step=0.01)
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, step=0.1)
    rh = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)

    if st.button("🔁 Run Simulation Step"):
        if image:
            img = Image.open(image).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)

            class_names = sorted([d for d in os.listdir("dataset/train") if os.path.isdir(os.path.join("dataset/train", d))])
            model = load_model(len(class_names))
            image_probs = predict_image(img, model)
            image_stage = class_names[np.argmax(image_probs)]

            sensor_stage = get_sensor_label(weight, temp, rh)
            sensor_probs = np.zeros(len(class_names))
            if sensor_stage in class_names:
                sensor_probs[class_names.index(sensor_stage)] = 1.0

            final_probs = 0.7 * image_probs + 0.3 * sensor_probs
            predicted_stage = class_names[np.argmax(final_probs)]

            control_signal = get_control_signal(predicted_stage, st.session_state.last_stage)
            st.session_state.last_stage = predicted_stage

            log_entry = {
                "Time": time.strftime("%H:%M:%S"),
                "Stage": predicted_stage,
                "Confidence": round(np.max(final_probs)*100, 2),
                "Control Action": control_signal
            }
            st.session_state.logs.append(log_entry)

            st.success(f"✅ Stage: {predicted_stage} | 🔧 Action: {control_signal}")
            st.markdown(f"**Confidence:** {log_entry['Confidence']}%")

            fig, ax = plt.subplots()
            ax.bar(class_names, final_probs, color="teal")
            ax.set_ylim([0, 1])
            ax.set_ylabel("Probability")
            ax.set_title("Class Probabilities")
            st.pyplot(fig)

    # Log Table
    if st.session_state.logs:
        st.subheader("📋 Drying Log")
        st.dataframe(pd.DataFrame(st.session_state.logs))

if __name__ == "__main__":
    main()
    