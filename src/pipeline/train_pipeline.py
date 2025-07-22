import sys
from dataclasses import dataclass
from tensorflow import keras
from src.component.model_definition import ModelBuilder,ModelConfig
from src.component.data_ingestion import DataIngestion
from src.exception import CustomException
import os

@dataclass
class TrainerConfig:
    epochs: int = 10
    learning_rate: float = 1e-5
    model_output_path: str = "artifacts/art_classifier_model.h5"

class ModelTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.data_ingestion = DataIngestion()
        model_config = ModelConfig()


        self.model = ModelBuilder(config=model_config).build_model()

    def train(self):
        try:
        
            train_ds = self.data_ingestion.get_train_dataset()
            val_ds = self.data_ingestion.get_val_dataset()

    
            self.model.compile(
                optimizer=keras.optimizers.RMSprop(learning_rate=self.config.learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            
            history = self.model.fit(
                train_ds,
                epochs=self.config.epochs,
                validation_data=val_ds
            )
            os.makedirs(os.path.dirname(self.config.model_output_path), exist_ok=True)

            
            self.model.save(self.config.model_output_path)
            print(f"Model training completed and saved to: {self.config.model_output_path}")

        except Exception as e:
            raise CustomException(e, sys)

def start_training():
    config = TrainerConfig()
    trainer = ModelTrainer(config=config)
    trainer.train()

if __name__ == "__main__":
    start_training()
