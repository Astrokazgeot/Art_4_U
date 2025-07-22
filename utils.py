import os
from PIL import Image

def validate_images_in_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                with Image.open(file_path) as img:
                    img.verify()  # verify if it's a valid image
            except Exception as e:
                print(f"❌ Invalid image: {file_path} — {e}")

# Check both training and validation sets
validate_images_in_directory("artifacts/training_set")
validate_images_in_directory("artifacts/validation_set")
