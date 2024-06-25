
import keras
from keras.layers import Conv2D, BatchNormalization, \
    MaxPool2D, GlobalMaxPool2D

from keras.layers import TimeDistributed, GRU, Dense, Dropout, ConvLSTM2D, Bidirectional, Flatten, Activation

class ModelManager():

    def __init__(self,params):
        print(params)
        self.shape=(params['nbframe'],) + params['size'] + (params['channels'],) # (5, 112, 112, 3)
        self.params = params
    
    def create_model(self):
        self.model_name=self.params['model']
        print("Creating model: "+self.model_name)
        self.__create_custom_model__(self.params)
        
    
    @staticmethod   
    def build_custom_convnet(shape=(112, 112, 3)):
        momentum = .9
        model = keras.Sequential()
        model.add(Conv2D(64, (3,3), input_shape=shape,
            padding='same', activation='relu'))
        model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(BatchNormalization(momentum=momentum))
        
        model.add(MaxPool2D())
        
        model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
        model.add(BatchNormalization(momentum=momentum))
        
        model.add(MaxPool2D())
        
        model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
        model.add(BatchNormalization(momentum=momentum))
        
        model.add(MaxPool2D())
        
        model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
        model.add(BatchNormalization(momentum=momentum))

        return model

    
    def __create_custom_model__(self,params):
        shape=self.shape
        # Create our convnet with (112, 112, 3) input shape
        if self.model_name=='custom':
            convnet = self.build_custom_convnet(shape[1:])
        elif self.model_name=='mobilenet':      #(8, 128, 128, 3)
            convnet = self.__create_mobilenet_model__(shape[1:])    #(128, 128, 3)
        
        model = keras.Sequential()
        model.add(TimeDistributed(convnet, input_shape=shape))

        #RNN witch ConvLSTM2D
        model.add(ConvLSTM2D(512, 1, 1, return_sequences=False, dropout=0.1))
        model.add(ConvLSTM2D(512, 1, 1, return_sequences=False, dropout=0.1))
        model.add(Activation('relu'))
        model.add(GlobalMaxPool2D())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(params['classes'], activation='softmax'))
        print("Summary of the final model:")
        model.summary()
        self.model=model

    def __create_mobilenet_model__(self,shape):
        print(shape)
        model = keras.applications.mobilenet.MobileNet(
            include_top=False,
            input_shape=shape,
            weights='imagenet') # usando os pesos da imagenet

        print('Trainable layers: ', self.params['trainable_layers'])
        trainable = self.params['trainable_layers']
        
        # Aqui é feito o processo de Fine-Tuning

        for layer in model.layers[:-trainable]:
            layer.trainable = False
        for layer in model.layers[-trainable:]:
            layer.trainable = True
        
        return model


        


    
    


