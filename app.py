import tkinter as tk
from tkinter import simpledialog
import cv2 as cv
import os
import PIL.Image, PIL.ImageTk
import model
import camera
import time

class App:
    def __init__(self, window=tk.Tk(), window_title="Camera Classifier"):
        self.window = window
        self.window.title(window_title)
        self.counters = [1, 1]

        self.model = model.Model()

        self.auto_predict = False

        # Initialize the camera with smaller resolution for faster frame capture
        self.camera = camera.Camera(width=640, height=480)

        self.init_gui()

        self.delay = 10  # Reduced delay for quicker frame updates
        time.sleep(1)  # Small delay to allow camera to warm up
        self.update()

        self.window.attributes("-topmost", True)
        self.window.mainloop()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text="Auto Prediction", width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.classname_one = simpledialog.askstring("Classname One", "Enter the name of the first class:", parent=self.window)
        self.classname_two = simpledialog.askstring("Classname Two", "Enter the name of the second class:", parent=self.window)

        self.btn_class_one = tk.Button(self.window, text=self.classname_one, width=50, command=lambda: self.save_for_class(1))
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_two = tk.Button(self.window, text=self.classname_two, width=50, command=lambda: self.save_for_class(2))
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text="Train Model", width=50, command=lambda: self.model.train_model(self.counters))
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predict", width=50, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="CLASS")
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if ret:
            if not os.path.exists(str(class_num)):
                os.mkdir(str(class_num))

            # Saving with lower quality for faster saving
            cv.imwrite(f'{class_num}/frame{self.counters[class_num-1]}.jpg', 
                       cv.cvtColor(frame, cv.COLOR_RGB2GRAY), 
                       [cv.IMWRITE_JPEG_QUALITY, 70])

            self.counters[class_num - 1] += 1

    def reset(self):
        for folder in ['1', '2']:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        self.counters = [1, 1]
        self.model = model.Model()
        self.class_label.config(text="CLASS")

    def update(self):
        if self.auto_predict:
            print(self.predict())

        ret, frame = self.camera.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def predict(self):
        ret, frame = self.camera.get_frame()
        if ret:
            prediction = self.model.predict(frame)
            if prediction == 1:
                self.class_label.config(text=self.classname_one)
                return self.classname_one
            elif prediction == 2:
                self.class_label.config(text=self.classname_two)
                return self.classname_two
