import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class ArtClassifierPredictor:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.image_size = (224, 224)

    def load_and_preprocess_image(self, img_path: str):
        img = image.load_img(img_path, target_size=self.image_size)
        img_array = image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict(self, img_path: str):
        img_array = self.load_and_preprocess_image(img_path)
        predictions = self.model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        return predicted_index, predictions[0]

