# ♻️ WasteLink Image Validation API

WasteLink is a FastAPI-based machine learning service designed to classify waste into five categories. This tool helps automate waste sorting to improve recycling efficiency.

## 🚀 Features
* **Custom ML Model:** Powered by a MobileNetV2 architecture fine-tuned on custom waste datasets.
* **FastAPI Backend:** High-performance asynchronous API for real-time predictions.
* **Cloud Deployment:** Hosted on Render for global accessibility.

## 📂 Project Structure
* `main.py`: The core FastAPI application logic.
* `wastelink_fixed.h5`: The trained TensorFlow model (H5 format for stability).
* `requirements.txt`: List of Python dependencies.

## 📊 Supported Categories
The model can identify the following:
1. **General**
2. **Metal**
3. **Organic**
4. **Paper**
5. **Plastic**

## 🛠️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/NjeriCodeCraft/validation.git]
