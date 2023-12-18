# Import the required libraries
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#Loading our model
model_res = tf.keras.models.load_model('resnet_model.h5')

#Title for our application
st.title('Malaria Affected Cell Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for prediction
    img_array = np.array(image)
    img_array = img_array/255
    img_array = np.expand_dims(img_array, axis=0)
    # Add preprocessing
    img_array = preprocess_input(img_array)

    # Make predictions
    prediction = model_res.predict(img_array)
    prediction = np.argmax(prediction, axis=1)
    
    # Display prediction result
    if prediction == 1:
        st.write('Prediction: Malaria Affected')
    else:
        st.write('Prediction: Unaffected')