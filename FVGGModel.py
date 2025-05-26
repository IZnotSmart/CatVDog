from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore

#Define VGG16 model
def defModel():
    #load the model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    #Freeze the layers of the base model
    for layer in model.layers:
        layer.trainable = False
    #Add custom layers on top of the base model
    x = Flatten()(model.layers[-1].output)
    x = Dense(128, activation='relu', kernel_initializer= 'he_uniform')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=model.inputs, outputs=x)
    #Compile the model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    model = defModel()
    #data generator
    datagen = ImageDataGenerator(featurewise_center=True)
    datagen.mean = [123.68, 116.779, 103.939]  # VGG16 mean pixel values
    #prepare iterators
    train_it = datagen.flow_from_directory('FdataCatDog/', target_size=(224, 224), class_mode='binary', batch_size=64)
    # fit model
    history = model.fit(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=1)
    # save model
    model.save('FModelCatDog.keras')


main()
print("end")