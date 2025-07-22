from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from src.services.predict import ArtPredictor
import uvicorn
import os

app = FastAPI(title="🎨 Art Classifier API")

# Enable CORS (for Streamlit or any frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ArtPredictor(
    model_path="artifacts/art_classifier_model.h5",
    class_names_path="artifacts/class_names.json"
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Art Classification API"}

@app.post("/predict/")
async def predict_art(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        temp_path = f"temp_{file.filename}"

        # Write the uploaded content to a temp file
        with open(temp_path, "wb") as f:
            f.write(contents)

        prediction = predictor.predict(temp_path)

        # Clean up the temp file
        os.remove(temp_path)

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
