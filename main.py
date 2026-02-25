import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf

app = FastAPI()

CATEGORIES = ['general', 'metal', 'organic', 'paper', 'plastic']
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        # Point to the NEW .keras file
        model = tf.keras.models.load_model('wastelink_v3.keras')
        print("✅ SUCCESS: Keras 3 Model Loaded!")
    except Exception as e:
        print(f"❌ Load Error: {e}")

@app.get("/")
def home():
    return {"message": "WasteLink API", "model_loaded": model is not None}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}
    
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB').resize((180, 180))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Keras 3 prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    
    return {
        "prediction": CATEGORIES[predicted_idx],
        "confidence": float(np.max(predictions[0]))
    }
