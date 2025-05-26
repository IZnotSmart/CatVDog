from keras.preprocessing.image  import load_img as keras_load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


def load_img(filename):
    # Load the image with target size (224, 224)
    img = keras_load_img(filename, target_size=(224, 224))
    # Convert the image to an array
    img = img_to_array(img)
    # Expand dimensions to match the model input shape
    img = img.reshape((1, 224, 224, 3))
    # Normalize the image array
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]

    return img


def predict():
    # Load the model
    model = load_model('FModelCatDog.keras')
    # Load and preprocess the image
    img = load_img('test.jpg')

    # Make a prediction
    prediction = model.predict(img)
    # Print the prediction result
    if prediction[0] > 0.5:
        print("The image is a dog.")
    else:
        print("The image is a cat.")


predict()
print("end")