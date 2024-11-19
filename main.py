from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import numpy as np
import tensorflow as tf
import streamlit as st

train_datagen = ImageDataGenerator(rescale=1./255)
train_dir = "weather_prediction/weather dataset/train/"
train_data = train_datagen.flow_from_directory(train_dir,
                                               batch_size=64,
                                               target_size=(224,224),
                                               class_mode='categorical')

def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def predict(img_path):
    model = tf.keras.models.load_model("weather_prediction/weather.keras")

    processed_image = load_and_preprocess_image(img_path)

    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)

    class_indices = train_data.class_indices
    class_labels = list(class_indices.keys())

    return class_labels[predicted_class[0]]


st.title("Weather Predictor")
img = st.file_uploader("Choose a file")
if img:
    st.image(img, caption="Uploaded Image", use_column_width=True)
    response = predict(img)
    st.write(response)