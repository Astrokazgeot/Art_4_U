import os
import sys
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.applications import ResNet50
from src.exception import CustomException

def main():
    try:
    
        conv_base = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )

        
        model = Sequential()
        model.add(conv_base)
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(8, activation='softmax'))  

        
        conv_base.trainable = True
        set_trainable = False

        for layer in conv_base.layers:
            if layer.name == "conv5_block1_out":  
                set_trainable = True
            layer.trainable = set_trainable

        
        train_ds = keras.utils.image_dataset_from_directory(
            directory="artifacts/training_set",
            labels="inferred",
            label_mode="int",
            batch_size=32,
            image_size=(224, 224)
        )

        val_ds = keras.utils.image_dataset_from_directory(
            directory="artifacts/validation_set",
            labels="inferred",
            label_mode="int",
            batch_size=32,
            image_size=(224, 224)
        )

    
        def process(image, label):
            image = tf.cast(image / 255.0, tf.float32)
            return image, label

        train_ds = train_ds.map(process).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.map(process).prefetch(buffer_size=tf.data.AUTOTUNE)

        
        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",  
            metrics=["accuracy"]
        )


        history = model.fit(
            train_ds,
            epochs=10,
            validation_data=val_ds
        )

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
