import cv2

class Camera:
    def __init__(self, width=640, height=480):
        self.cap = cv2.VideoCapture(0)
        self.width = width
        self.height = height

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, None

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
