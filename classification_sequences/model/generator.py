import os 
import os
import glob
import keras
from keras_video import VideoFrameGenerator

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
    split=None, 
    shuffle=False,
    batch_size=batch_size,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    transformation=None,
    use_frame_cache=True)