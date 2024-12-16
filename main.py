from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from api.predict import predict

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Model API is running!"}

@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    # Read and preprocess the uploaded image
    image = Image.open(file.file).resize((128, 128))  # Adjust size for your model
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    
    # Run prediction
    result = predict(image_array)
    return {"prediction": result}
