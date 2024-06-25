from model.eval import ModelEval


import os
import glob
import keras
from keras_video import VideoFrameGenerator
import keras_video.utils
import matplotlib.pyplot as plt
import numpy as np
import cv2


from model.trainer import ModelTrainer

classes = [i.split(os.path.sep)[1] for i in glob.glob('videos/*')]
classes.sort()

SIZE = (112, 112)
CHANNELS = 3
numero_de_frames = 5
batch_size = 8


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
    transformation=None,
    use_frame_cache=True)

validation_generator = train_generator.get_validation_generator()

model_path = 'chkp/best_model_0.h5'
model = keras.models.load_model(model_path)
evaluator = ModelEval(model, validation_generator)
#evaluator.evaluate_with_generator()
evaluator.plot_and_predict_with_generator()