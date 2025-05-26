import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

def defModel():
    # load the VGG16 model
    base_model = VGG16(include_top=False, input_shape=(200, 200, 3))
    # freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False
    # add custom layers on top of the base model
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    # compile the model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    defModel()
    print("aaa")

main()
print("end")