import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from src.database.operations import insert_artwork
from src.services.predict import ArtPredictor

router = APIRouter()
predictor = ArtPredictor(model_path="src/models/art_model.h5")

UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-art/")
async def upload_artwork(
    title: str = Form(...),
    artist_name: str = Form(...),
    style: str = Form(...),
    file: UploadFile = File(...)
):
    try:
       
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as out_file:
            content = await file.read()
            out_file.write(content)


        prediction_result = predictor.predict(file_path)

        
        insert_artwork(
            title=title,
            artist_name=artist_name,
            style=style,
            prediction_result=prediction_result,
            image_path=file_path
        )

        return {
            "message": "Artwork uploaded and classified successfully.",
            "prediction": prediction_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload artwork: {str(e)}")
