import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from src.exception import CustomException

import json
import os

class ArtPredictor:
    def __init__(self, model_path: str, class_names_path: str = "artifacts/class_names.json"):
        try:
            self.model = load_model(model_path)
            self.image_size = (224, 224)

            if not os.path.exists(class_names_path):
                raise FileNotFoundError(f"Missing class names file: {class_names_path}")

            with open(class_names_path, "r") as f:
                self.class_names = json.load(f)

        except Exception as e:
            raise CustomException(f"Failed to initialize predictor: {e}", sys)


    def predict(self, image_path: str) -> str:
        try:
            img = image.load_img(image_path, target_size=self.image_size)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = self.model.predict(img_array)
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_label = self.class_names[predicted_index]

            return predicted_label

        except Exception as e:
            raise CustomException(f"Prediction failed: {e}", sys)


