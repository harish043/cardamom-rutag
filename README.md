# 🌿 Cardamom Drying Optimization System

An intelligent multimodal system for monitoring and optimizing the drying process of cardamom using computer vision and sensor fusion. Developed during a research internship at **RuTAG, IIT Madras**.

## 🚀 Overview

This application classifies the dryness stage of cardamom in real-time using a fine-tuned **EfficientNet-B0** model and IoT sensor inputs (temperature, humidity, and weight). It simulates a closed-loop heat pump control system by logging stage transitions and issuing drying control signals.

### 🔧 Features

- 🔍 Image-based dryness classification (raw, partially_dried, mostly_dried, fully_dried)
- 🌡️ Real-time sensor fusion with temperature, humidity, and weight
- 🧠 Multimodal label fusion using rule-based heuristics
- 🔁 Dynamic simulation of heat pump actuation logic
- 📊 Interactive and responsive UI via **Streamlit**
- 📈 Auto-generated drying logs and confidence plots

---

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Model**: PyTorch (EfficientNet-B0)
- **Sensor Fusion**: Weighted rule-based integration
- **Control Logic**: State transition logic for adaptive drying
- **Visualization**: Matplotlib, Streamlit

---

## 🗂 Folder Structure

cardamom-rutag/
├── app.py # Main Streamlit app
├── downloader.py # Optional GDrive model downloader
├── models/
│ └── efficientnet_cardamom_best.pth # Model file (excluded from repo)
├── dataset/
│ └── train/ # Folder containing class directories (for label mapping)
├── requirements.txt
├── README.md


---

## 📈 Dryness Stages

The classifier predicts the following drying stages:
- `raw`
- `partially_dried`
- `mostly_dried`
- `fully_dried`

---

## 🔁 Control System Logic

| Situation                     | Action               |
|------------------------------|----------------------|
| Initial state                | START                |
| Stage not progressing (raw)  | INCREASE_TEMP        |
| Stage not progressing        | STABLE / MAINTAIN    |
| Stage regressing             | INCREASE_TEMP        |
| Normal stage transition      | MAINTAIN             |
| Nearing final stage          | REDUCE_TEMP          |
| Fully dried                  | STOP_HEAT            |

---

## 📦 Model Download (Google Drive)

To keep the repo lightweight, the model is stored on Google Drive.

You can download it using:

```bash
python downloader.py

##To run:

pip install -r requirements.txt
streamlit run app.py

##🤝 Acknowledgments

    Internship under RuTAG, IIT Madras

    Field insights from agricultural experts

    Mentorship and guidance from academic supervisors

