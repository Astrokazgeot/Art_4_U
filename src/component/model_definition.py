from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import ResNet50

@dataclass
class ModelConfig:
    input_shape: tuple = (224, 224, 3)
    num_classes: int = 8
    fine_tune_from_layer: str = "conv5_block1_out"

class ModelBuilder:
    def __init__(self, config: ModelConfig):
        self.config = config

    def build_model(self):
        conv_base = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.config.input_shape
        )

        model = Sequential([
            conv_base,
            Flatten(),
            Dense(256, activation='relu'),
            Dense(self.config.num_classes, activation='softmax')
        ])

        
        conv_base.trainable = True
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == self.config.fine_tune_from_layer:
                set_trainable = True
            layer.trainable = set_trainable

        return model
