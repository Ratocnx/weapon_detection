import numpy as np
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import cv2
import matplotlib.pyplot as plt

#Classe para avaliar o modelo de classificação CNN+RNN com o uso de um generator

class ModelEval:
    def __init__(self, model, generator=None):
        self.model = model
        self.generator = generator

    def evaluate_with_generator(self):
        #Esta função avalia o modelo com o generator passado e printa as métricas Accuracy, Precision, Recall, F1 e 
        #a matriz de confusão

        total_elements = len(self.generator) * self.generator.batch_size
        print("Total elements in generator: ", total_elements)

        y_true =[]
        y_pred = []

        # Percorre o generator e faz a predição para cada batch
        for i in range(len(self.generator)):
            X, y = next(self.generator)
            temp_y_pred = self.model.predict(X)
            temp_y_pred = np.argmax(temp_y_pred, axis=1)
            temp_y_true = np.argmax(y, axis=1)
            y_pred.extend(temp_y_pred)
            y_true.extend(temp_y_true)
        self.calculate_metrics(y_true, y_pred)


    @staticmethod
    def calculate_metrics(y_true, y_pred):
        #Esta função calcula as métricas Accuracy, Precision, Recall, F1 e a matriz de confusão

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred)
        
        #Print das métricas
        print("\n__________ Metrics__________\n")
        print("Accuracy: {:.2f}".format(accuracy))
        print("Precision: {:.2f}".format(precision))
        print("Recall: {:.2f}".format(recall))
        print("F1: {:.2f}".format(f1))
        print("Confusion Matrix: \n", cm)

    def plot_and_predict_with_generator(self):
        #Esta função plota as imagens do generator junto com a predição do modelo
        X, y = next(self.generator)
        y_pred = self.model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y, axis=1)
        classes= self.generator.classes

        # Percorre o batch e plota as sequências de imagens
        for i in range(X.shape[0]):
            elongated_image = np.zeros((X.shape[2], X.shape[2]*X.shape[1], X.shape[4]), dtype=np.uint8)
            for j in range(X.shape[1]):
                frame = X[i, j, :, :, :]
                frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255
                frame = frame.astype(np.uint8)
                elongated_image[:, j*X.shape[2]:(j+1)*X.shape[2], :] = frame
            plt.imshow(elongated_image.astype(int))
            plt.title("True: {}  ---------  Pred: {}".format(classes[y_true[i]], classes[y_pred[i]]))
            plt.show()
            





                






            
