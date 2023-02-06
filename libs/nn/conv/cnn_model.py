from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
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

        # INPUT => [CONV => RELU => CONV => RELU => POOL] * 3 => [FC => RELU] * 2 => FC
        model = Sequential(
            [
                Conv2D(16, (3, 3),padding="same",input_shape=input_shape,activation = 'relu'),
                MaxPooling2D(pool_size=2),
                Conv2D(32, (3, 3),padding="same",activation = 'relu'),
                MaxPooling2D(pool_size=2),

                Conv2D(64, (3, 3),padding="same",activation = 'relu'),
                MaxPooling2D(pool_size=2),      
                Dropout(0.3),

                Dense(units=128, activation = 'relu'),
                Dense(units=256, activation = 'relu'),
                Dropout(0.3),

                Flatten(),
                Dense(units=512, activation = 'relu'),
                Dense(1, activation = 'sigmoid'),
            ]
        )

        print(model.summary())
        return model
