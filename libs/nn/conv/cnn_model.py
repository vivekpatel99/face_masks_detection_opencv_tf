from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K


class CnnModel:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # channels last
        input_shape = (height, width, depth)

        # if we are using channels first, update the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        model = Sequential()

        model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
 
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
     
        model.add(Dense(1024))

        model.add(Dense(64))

        model.add(Dense(classes, activation='sigmoid'))
        return model

    @staticmethod
    def normalize(dataset):
        normalization_layer = Rescaling(1./255)
        return dataset.map(lambda x, y: (normalization_layer(x), y))