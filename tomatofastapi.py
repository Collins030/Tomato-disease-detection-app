from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Allow frontend to communicate with backend
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL = tf.keras.models.load_model("C:/Users/User/OneDrive/Documents/tomato_disease_detection/model/tomato_model.h5")

# Define class names and updated recommendations
CLASS_NAMES = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

RECOMMENDATIONS = {
    "Tomato___Bacterial_spot": {
        "solution": "Use copper-based bactericides such as Mancozeb + Copper Hydroxide or Streptomycin sulfate.",
        "chemicals": [
            {
                "name": "Mancozeb + Copper Hydroxide",
                "dosage": "2.5g per liter of water",
                "procedure": "Spray weekly during humid conditions.",
                "image_url": "https://example.com/images/mancozeb_copper_hydroxide.jpg"
            }
        ]
    },
    "Tomato___Early_blight": {
        "solution": "Apply fungicides containing Chlorothalonil or Azoxystrobin.",
        "chemicals": [
            {
                "name": "Chlorothalonil",
                "dosage": "2.5g per liter of water",
                "procedure": "Spray at first sign of disease and repeat every 7–10 days.",
                "image_url": "https://example.com/images/chlorothalonil.jpg"
            }
        ]
    },
    "Tomato___Late_blight": {
        "solution": "Use Cymoxanil or Mefenoxam fungicides.",
        "chemicals": [
            {
                "name": "Cymoxanil",
                "dosage": "3g per liter of water",
                "procedure": "Apply every 5–7 days during wet weather conditions.",
                "image_url": "https://example.com/images/cymoxanil.jpg"
            }
        ]
    },
    "Tomato___Leaf_Mold": {
        "solution": "Use Copper Hydroxide or Mancozeb fungicides.",
        "chemicals": [
            {
                "name": "Copper Hydroxide",
                "dosage": "2g per liter of water",
                "procedure": "Apply during high humidity periods.",
                "image_url": "https://example.com/images/copper_hydroxide.jpg"
            }
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "solution": "Apply Chlorothalonil or Copper Hydroxide fungicides.",
        "chemicals": [
            {
                "name": "Chlorothalonil",
                "dosage": "3g per liter of water",
                "procedure": "Spray every 7–14 days.",
                "image_url": "https://example.com/images/chlorothalonil.jpg"
            }
        ]
    },
    "Tomato___Spider_mites": {
        "solution": "Apply miticides such as Abamectin or Bifenazate.",
        "chemicals": [
            {
                "name": "Abamectin",
                "dosage": "1.5ml per liter of water",
                "procedure": "Spray under leaves where mites are found.",
                "image_url": "https://example.com/images/abamectin.jpg"
            }
        ]
    },
    "Tomato___Target_Spot": {
        "solution": "Use Difenoconazole or Azoxystrobin fungicides.",
        "chemicals": [
            {
                "name": "Azoxystrobin",
                "dosage": "2ml per liter of water",
                "procedure": "Apply at disease onset and repeat every 10–14 days.",
                "image_url": "https://example.com/images/azoxystrobin.jpg"
            }
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "solution": "No direct chemical control; manage vectors (whiteflies) using Imidacloprid.",
        "chemicals": [
            {
                "name": "Imidacloprid",
                "dosage": "1ml per liter of water",
                "procedure": "Apply systemically to control whiteflies.",
                "image_url": "https://example.com/images/imidacloprid.jpg"
            }
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "solution": "No direct treatment; use resistant varieties and sanitation.",
        "chemicals": []
    },
    "Tomato___healthy": {
        "solution": "No action required. Maintain crop monitoring and healthy practices.",
        "chemicals": []
    }
}

# Helper function to read and process the uploaded image
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    # Perform prediction
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Include recommendations in the response
    recommendations = RECOMMENDATIONS[predicted_class]

    return {
        "class": predicted_class,
        "confidence": float(confidence),
        "recommendation": recommendations
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
