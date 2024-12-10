import numpy as np
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier
import os

class Model:
    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.X = []
        self.y = []

    def preprocess_image(self, frame):
        if frame is None:
            print("Error: Frame is empty.")
            return None
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        resized = cv.resize(gray, (64, 64)).flatten()
        return resized

    def train_model(self, counters):
        # Load images from class 1 and 2 folders
        for class_num in [1, 2]:
            folder = str(class_num)
            for file in os.listdir(folder):
                img_path = os.path.join(folder, file)
                img = cv.imread(img_path)
                processed_img = self.preprocess_image(img)
                if processed_img is not None:
                    self.X.append(processed_img)
                    self.y.append(class_num)
        
        # Train KNN model
        if self.X and self.y:
            self.knn.fit(self.X, self.y)
            print(f"Training completed with {len(self.X)} samples.")
        else:
            print("No training data available.")

    def predict(self, frame):
        processed_frame = self.preprocess_image(frame)
        if processed_frame is not None:
            prediction = self.knn.predict([processed_frame])
            return prediction[0]
        else:
            print("Prediction failed due to empty frame.")
            return None
