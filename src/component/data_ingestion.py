import os
import sys
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import layers
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_dir = os.path.join("artifacts", "training_set")
    val_dir = os.path.join("artifacts", "validation_set") 
    img_height = 224
    img_width = 224
    batch_size = 32

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def load_dataset(self, directory):
        try:
            dataset = tf.keras.utils.image_dataset_from_directory(
                directory,
                image_size=(self.config.img_height, self.config.img_width),
                batch_size=self.config.batch_size,
                shuffle=True
            )
            normalization_layer = layers.Rescaling(1./255)
            dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
            return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        except Exception as e:
            raise CustomException(e, sys)

    def get_train_dataset(self):
        return self.load_dataset(self.config.train_dir)

    def get_val_dataset(self):  
        return self.load_dataset(self.config.val_dir)

