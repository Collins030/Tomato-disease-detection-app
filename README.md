# 🍅 Tomato Disease Detection App

Welcome to the **Tomato Disease Detection App**, an end-to-end machine learning project designed to identify tomato leaf diseases from images and provide actionable recommendations. This app combines a trained TensorFlow model, a FastAPI backend, and a React-based frontend to deliver predictions directly to the user.

---

## 🌟 Features

- **Model Training**: Build and train a TensorFlow model to classify tomato leaf diseases with high accuracy.
- **FastAPI Backend**: Hosts the model locally, processes image uploads, and serves predictions to the frontend.
- **Interactive Frontend**: React app that allows users to upload images and view real-time predictions with actionable advice.
- **Actionable Recommendations**: Provides disease-specific treatments and farming tips.

---

## 📂 Project Structure

| File/Folder         | Description                                                                                     |
|---------------------|-------------------------------------------------------------------------------------------------|
| `tomato.py`         | Script to train and save the TensorFlow model for detecting tomato diseases.                    |
| `tomatofastapi.py`  | FastAPI application that serves the trained model and provides predictions for uploaded images. |
| `frontend`          | React app that interacts with the FastAPI backend to display predictions and recommendations.  |
| `model`             | Directory to store the trained model (`tomato_model.h5`).                                       |
| `README.md`         | Comprehensive guide to understanding and using the project.                                    |

---

## 🛠️ Setup Instructions

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Collins030/Tomato-disease-detection-app.git
cd Tomato-disease-detection-app

2. Train the Model (Optional)
If you want to retrain the model, use the tomato.py script. Ensure you have the required dataset and TensorFlow installed.

bash
Copy code
python tomato.py
3. Run the FastAPI Backend
Start the backend using uvicorn to serve the model locally.

bash
Copy code
uvicorn tomatofastapi:app --reload
By default, the FastAPI server runs at http://localhost:8000.

4. Start the Frontend
Navigate to the frontend directory and start the React app:

bash
Copy code
cd frontend
npm install
npm start
The React app will run at http://localhost:3000.

🚀 How It Works
Model Training:

The tomato.py script preprocesses the dataset, trains a CNN model using TensorFlow, and saves the trained model as tomato_model.h5.
FastAPI Backend:

The tomatofastapi.py script loads the trained model and provides an endpoint (/predict) for image-based predictions.
Users can upload images via the React app, and the backend responds with the predicted class and recommendations.
Frontend:

The React app interacts with the FastAPI backend to allow users to upload images and view predictions in an intuitive interface.
🖼️ Demo
Here’s a visual walkthrough of the app in action:

1. Upload Image
Upload a tomato leaf image via the React app.



2. View Prediction
See the predicted class (e.g., Tomato___Bacterial_spot) and confidence score.



3. Get Recommendations
Receive tailored advice, including fungicide recommendations and usage instructions.



🧑‍💻 Technologies Used
Backend: Python, TensorFlow, FastAPI
Frontend: React.js
Database: None (uses local file storage for model and static assets)
💡 Inspiration
This project is inspired by the Potato Disease Detection App, which employs similar techniques for detecting diseases in potato plants.

📜 License
This project is open-source and available under the MIT License.

🌱 Future Enhancements
Integration with cloud storage for model hosting and scalability.
Support for additional crops and diseases.
Deployment of the app for public use.
Feel free to contribute by raising issues or submitting pull requests!
