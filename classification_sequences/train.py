import os
import glob
import keras
from keras_video import VideoFrameGenerator
import keras_video.utils
import matplotlib.pyplot as plt
import numpy as np
import cv2

#python train.py
from model.trainer import ModelTrainer
from model.eval import ModelEval

classes = [i.split(os.path.sep)[1] for i in glob.glob('videos/*')]
classes.sort()

SIZE = (512, 512)
CHANNELS = 3
numero_de_frames = 5
batch_size = 3
trainable_layers = 4       # 5, 20, 40, 80, 250

glob_pattern='videos/{classname}/*.avi'

data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)

train_generator = VideoFrameGenerator(
    classes=classes, 
    glob_pattern=glob_pattern,
    nb_frames=numero_de_frames,
    split=.33, 
    shuffle=True,
    batch_size=batch_size,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    transformation=data_aug,
    use_frame_cache=False)

validation_generator = train_generator.get_validation_generator()

Z,y=next(validation_generator)  #z = (8, 5, 112, 112, 3) 
print(Z.shape)
print(y.shape)

#plot the first image of the first sequence
# plt.imshow(Z[0,0,:,:,:])
# plt.show()
# keras_video.utils.show_sample(validation_generator)

params = {
    'classes': len(classes),
    'nbout': len(classes),
    'size': SIZE,
    'epochs': 50, 
    'channels': CHANNELS,
    'nbframe': numero_de_frames,
    'model': 'mobilenet',
    'trainable_layers': trainable_layers,
}

trainer=ModelTrainer(params)
trainer.train(train_generator,validation_generator)

evaluator = ModelEval(trainer.model, validation_generator)
evaluator.evaluate_with_generator()

#Carregar o modelo

# model = keras.models.load_model('best_model')

# #Procurar uma biblioteca pra usar a webcam

# frames = []

# classes = ['boxing', 'handclapping', 'shoot_gun']
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = cv2.resize(frame, (112, 112))
#     frames.append(frame)
#     if len(frames) == 5:
#         frames = np.array(frames)
#         frames = np.expand_dims(frames, axis=0) # (1, 5, 112, 112, 3)
#         prediction = model.predict(frames)  # (1, 3)    (0.1, 0.2, 0.7) 3
#         prediction = np.argmax(prediction)
#         print(classes[prediction])  #shoot_gun
#         '''Se o modelo consegue prededir o que a pessoa está fazendo,
#         passar o ultimo frame da sequencia pro modelo 2'''
#         modelo_2.predict(frames[:,-1,:,:,:])
#         break