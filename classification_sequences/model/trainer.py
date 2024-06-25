
from model.models import ModelManager
import keras
from keras.losses import categorical_crossentropy


class ModelTrainer():

    def __init__(self, params):
        self.params = params
        self.create_model_architecture()

    def create_model_architecture(self):
        self.model_manager = ModelManager(self.params)
        self.model_manager.create_model()
        self.model=self.model_manager.model

        return self.model

    def compile_model(self):
        optimizer = keras.optimizers.Adam(0.001)
        self.model.compile(
        optimizer,
        loss=categorical_crossentropy,
        metrics=['accuracy']
        )
    
    def train(self,train_generator,val_generator):
        self.compile_model()

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(verbose=1,patience=4),
            keras.callbacks.ModelCheckpoint(
            'chkp/best_model.h5',
            verbose=1,
            save_best_only=True,
            save_weights_only=False
            ),
            keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1
            )

        ]

        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=self.params['epochs'],
            verbose=1,
            callbacks=callbacks
        )

    def save_model(self):
        self.model.save('best_model.h5')