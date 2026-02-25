import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf

app = FastAPI()

# Your specific categories from Colab
CATEGORIES = ['general', 'metal', 'organic', 'paper', 'plastic']

model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        # This loads your new .h5 file
        model = tf.keras.models.load_model('wastelink_fixed.h5', compile=False)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model failed to load: {e}")

@app.get("/")
def home():
    return {"message": "WasteLink API is Live", "model_loaded": model is not None}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Process image
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB').resize((180, 180))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Normalize (Scaling used in Colab)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    
    return {
        "prediction": CATEGORIES[predicted_idx],
        "confidence": float(np.max(predictions[0])),
        "status": "success"
    }