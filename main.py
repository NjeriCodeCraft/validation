import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from tensorflow.keras import layers, models

app = FastAPI()

CATEGORIES = ['general', 'metal', 'organic', 'paper', 'plastic']
model = None

def build_model_structure():
    # We build the body manually so Render doesn't get confused
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(180, 180, 3), 
        include_top=False, 
        weights=None 
    )
    
    m = models.Sequential([
        layers.Input(shape=(180, 180, 3)),
        layers.Rescaling(1./127.5, offset=-1),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(len(CATEGORIES), activation='softmax')
    ])
    return m

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = build_model_structure()
        # Now we just pour the weights into the body
        model.load_weights('model_weights.weights.h5')
        print("✅ SUCCESS: Model is ready!")
    except Exception as e:
        print(f"❌ Error: {e}")

@app.get("/")
def home():
    return {"model_loaded": model is not None}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None: return {"error": "Model not ready"}
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB').resize((180, 180))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model(img_array, training=False)
    predicted_idx = np.argmax(predictions.numpy()[0])
    
    return {
        "prediction": CATEGORIES[predicted_idx],
        "confidence": float(np.max(predictions.numpy()[0]))
    }
