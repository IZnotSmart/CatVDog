import sys
from matplotlib import pyplot
from keras.utils import to_categorical
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

#Summarize history for accuracy and loss
def summarise(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.legend()
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.legend()
    #save the plots
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    # show the figure
    pyplot.show()
    pyplot.close()

def main():
    model = defModel()
    #data generator
    datagen = ImageDataGenerator(featurewise_center=True)
    datagen.mean = [123.68, 116.779, 103.939]  # VGG16 mean pixel values
    #prepare iterators
    train_it = datagen.flow_from_directory('dataCatDog/train/', target_size=(224, 224), class_mode='binary', batch_size=64)
    test_it = datagen.flow_from_directory('dataCatDog/test/', target_size=(224, 224), class_mode='binary', batch_size=64)    
    # fit model
    history = model.fit(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=1)
    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print ('> %.3f' % (acc * 100.0))
    # summarize history for accuracy and loss
    summarise(history)

main()
print("end")