import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from src.database.operations import insert_artwork
from src.services.predict import ArtPredictor

router = APIRouter()
predictor = ArtPredictor(model_path="artifacts/art_classifier_model.h5")

UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/predict/")
async def upload_artwork(
    file: UploadFile = File(...)
):
    try:
      
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as out_file:
            content = await file.read()
            out_file.write(content)

        prediction_result = predictor.predict(file_path)

        insert_artwork(
            title="Unknown Title",
            artist_name="Unknown Artist",
            style="Unknown Style",
            prediction_result=prediction_result,
            image_path=file_path
        )

        return {
            "message": "Artwork uploaded and classified successfully.",
            "prediction": prediction_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload artwork: {str(e)}")
