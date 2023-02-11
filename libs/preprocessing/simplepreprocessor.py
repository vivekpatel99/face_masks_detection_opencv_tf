import cv2
import numpy as np

class SimplePreprocessor:
    def __init__(self, width:int, height:int, inter=cv2.INTER_AREA) -> None:
        """
        Store the target image width, height and interpolation method
        used when resizing
        """
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image:np.array) -> np.array:
        """ 
        resize the image to a fixed size, ignoring the aspect ratio
        """
        return cv2.resize(
            image, 
            (self.width, self.height),
            interpolation=self.inter)