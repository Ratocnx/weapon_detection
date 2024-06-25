import numpy as np
import keras
import glob
import cv2
import os
import PIL

from keras_video import VideoFrameGenerator
from ultralytics import YOLO

'''
> python pipeline.py
'''

#Load models
model_path = 'chkp/best_model_2_256x256.h5'
rnn_model = keras.models.load_model(model_path)

detection_model = YOLO('YOLO/bestnm.pt')

def predict_with_detection_model(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #plot_image(image, 'image')
    results = detection_model(image, show=False, conf=0.4, save=False)
    return results

def plot_image(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Load generator
classes = [i.split(os.path.sep)[1] for i in glob.glob('videos/*')]
classes.sort()
print(f'Classes: {classes}')
#get the index of 'shoot_gun' class
#shoot_gun_idx = classes.index('shoot_gun')
shoot_gun_idx = 5
print(f'shoot_gun index: {shoot_gun_idx}')
SIZE = (256, 256)
CHANNELS = 3
numero_de_frames = 5
batch_size = 1
val_split = 0.50

glob_pattern='videos/{classname}/*.avi'

generator = VideoFrameGenerator(
    classes=classes, 
    glob_pattern=glob_pattern,
    nb_frames=numero_de_frames,
    split=val_split,
    shuffle=False,
    batch_size=batch_size,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    transformation=None,
    use_frame_cache=True)

validation_generator = generator.get_validation_generator()

generator_2 = VideoFrameGenerator(
    classes=classes, 
    glob_pattern=glob_pattern,
    nb_frames=10,
    split=val_split,
    shuffle=False,
    batch_size=batch_size,
    target_shape=(550, 550),
    nb_channel=CHANNELS,
    transformation=None,
    use_frame_cache=True)

validation_generator_2 = generator_2.get_validation_generator()

#Get y_true from generator and get the positive values for shoot_gun, append 1 if argmax is 5 and 0 otherwise
y_true = []
for i in range(len(validation_generator)):
    X, y = validation_generator_2.__getitem__(i, return_single=False)
    for j in range(len(y)):
        if np.argmax(y[j]) == shoot_gun_idx:
            y_true.append(1)
        else:
            y_true.append(0)

#print(f'y_true: {y_true}')

#Predict
rnn_preds = rnn_model.predict(validation_generator)
print(f'RNN preds: {rnn_preds}')
shoot_gun = []
yoloshoot_gun = []
shoot_gun_true = []

for idx,pred in enumerate(rnn_preds):
    if np.argmax(pred) == shoot_gun_idx:
        shoot_gun_true.append(1)
        shoot_gun.append(1)
    else:
        shoot_gun.append(0)

#print(f'RNN preds: {shoot_gun}')

#YOLO Preds for shoot_gun, if at least one frame has a score > 0, append 1, otherwise append 0
for i in range(len(validation_generator_2)):
    X, y = validation_generator_2.__getitem__(i, return_single=False)
    for j in range(X.shape[1]):
        current_frame = X[0][j]
        current_frame = (current_frame * 255).astype(np.uint8)
        results = predict_with_detection_model(current_frame)
        scores = results[0].boxes.conf
        if len(scores) == 0:
            continue
        print(f'Score: {scores[0]}')
        if scores[0] > 0:
            yoloshoot_gun.append(1)
            break
    else:
        yoloshoot_gun.append(0)

#print(f'Yolo preds: {yoloshoot_gun}')

#Firts get the good predictions for RNN
good_preds_rnn = 0
bad_preds_rnn = 0
for i in range(len(shoot_gun)):
    if shoot_gun[i] == y_true[i]:
        good_preds_rnn += 1
    else:
        bad_preds_rnn += 1

#Then get the good predictions for YOLO
good_preds_yolo = 0
bad_preds_yolo = 0
for i in range(len(yoloshoot_gun)):
    if yoloshoot_gun[i] == y_true[i]:
        good_preds_yolo += 1
    else:
        bad_preds_yolo += 1
#Then get the good predictions for both by multiplying the two lists
yoloshoot_gun_m = list(np.array(yoloshoot_gun) * np.array(shoot_gun))
good_preds_both = 0
bad_preds_both = 0
for i in range(len(shoot_gun)):
    if shoot_gun[i] == yoloshoot_gun_m[i] == y_true[i]:
        good_preds_both += 1
    else:
        bad_preds_both += 1

#Then get the good predictions for both by applying an OR operation to the two lists and not '+' operation
yoloshoot_gun_s = []
for i in range(len(shoot_gun)):
    if shoot_gun[i] == 1 or yoloshoot_gun[i] == 1:
        yoloshoot_gun_s.append(1)
    else:
        yoloshoot_gun_s.append(0)

good_preds_both_s = 0
bad_preds_both_s = 0
for i in range(len(shoot_gun)):
    if yoloshoot_gun_s[i] == y_true[i]:
        good_preds_both_s += 1
    else:
        bad_preds_both_s += 1

print(f'Y_true: {y_true}')
print(f'RNN preds: {shoot_gun}')
print(f'YOLO preds: {yoloshoot_gun}')
print(f'YOLO preds mult: {yoloshoot_gun_m}')
print(f'YOLO preds sum: {yoloshoot_gun_s}')

print(f'Good preds RNN: {good_preds_rnn}')
print(f'Bad preds RNN: {bad_preds_rnn}')
FP = 0
FN = 0
for i in range(len(shoot_gun)):
    if shoot_gun[i] == 1 and y_true[i] == 0:
        FP += 1
    if shoot_gun[i] == 0 and y_true[i] == 1:
        FN += 1
# print(f'False Positives: {FP}')
# print(f'False Negatives: {FN}\n')

print(f'Good preds YOLO: {good_preds_yolo}')
print(f'Bad preds YOLO: {bad_preds_yolo}')
FP = 0
FN = 0
for i in range(len(shoot_gun)):
    if yoloshoot_gun[i] == 1 and y_true[i] == 0:
        FP += 1
    if yoloshoot_gun[i] == 0 and y_true[i] == 1:
        FN += 1
# print(f'False Positives: {FP}')
# print(f'False Negatives: {FN}\n')

print(f'Good preds Both mult: {good_preds_both}')
print(f'Bad preds Both mult: {bad_preds_both}')
#For this model (Both mult), show the quantity of False Positives and False Negatives
FP = 0
FN = 0
for i in range(len(shoot_gun)):
    if yoloshoot_gun_m[i] == 1 and y_true[i] == 0:
        FP += 1
    if yoloshoot_gun_m[i] == 0 and y_true[i] == 1:
        FN += 1
# print(f'False Positives: {FP}')
# print(f'False Negatives: {FN}\n')

print(f'Good preds Both sum: {good_preds_both_s}')
print(f'Bad preds Both sum: {bad_preds_both_s}')














#Use webcam
# cap = cv2.VideoCapture(0)

# frames = []
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Webcam error")
#         break
#     frame = cv2.resize(frame, SIZE)
#     frames.append(frame)
#     if len(frames) == numero_de_frames:
#         X = np.array(frames)
#         X = np.expand_dims(X, axis=0)
#         y_pred = rnn_model.predict(X)
#         y_pred = np.argmax(y_pred, axis=1)
#         print(classes[y_pred[0]])
#         frames = []
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
