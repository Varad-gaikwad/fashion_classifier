import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('fashion_classifier.keras')

classes = [
    "Ankle Boot",
    "Bag",
    "Coat",
    "Dress",
    "Pullover",
    "Sandal",
    "Shirt",
    "Sneaker",
    "T-shirt",
    "Trouser"
]

st.title("Fashion MNIST Classifier")

st.write("Upload a clothing image to predict its category.")
st.write("The model can classify the following categories:")
st.write("Ankle Boot, Bag, Coat, Dress, Pullover, Sandal, Shirt, Sneaker, T-shirt, Trouser")

upload=st.file_uploader("Upload an image to predict its catergory", type=["jpg", "jpeg", "png"])

if upload is not None:
    image= Image.open(upload)
    st.image(image, caption="Uploaded Image", width=250)

    #Process the image and covert it to grayscale and resize it to 28x28 pixels
    image = image.convert("L")
    image = image.resize((28, 28))

    #Convert the image to a numpy array
    image_array = np.array(image).astype("float32")

    #Reshape the image to match the input shape of the model
    image_array= image_array.reshape(1, 28, 28)

    prediction = model.predict(image_array)

    predicted_class= np.argmax(prediction)
    probabilities= prediction[0]

    confidence= probabilities[predicted_class]



    st.write(f"Predicted Class: {classes[predicted_class]}")
    st.write(f"Confidence: {confidence*100:.2f}%")
    

    
