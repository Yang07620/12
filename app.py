import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Cargar modelo
model = tf.keras.models.load_model('model/model.h5')

st.title("Clasificador de Dígitos (1 a 9)")
st.write("Sube una imagen de un dígito escrito a mano (1–9).")

uploaded_file = st.file_uploader("Selecciona una imagen...", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    img = image.convert('L')               # Convertir a escala de grises
    img = ImageOps.invert(img)             # Invertir blanco/negro
    img = img.resize((28, 28))             # Redimensionar
    img = np.array(img) / 255.0            # Normalizar
    return img

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_container_width=True)  # Aquí se hace el cambio

    img = preprocess_image(image)
    pred = model.predict(np.expand_dims(img, axis=0))[0]

    st.subheader(f"Número predicho: **{np.argmax(pred)+1}**")
    
    st.subheader("Probabilidades:")
    for i, p in enumerate(pred):
        st.write(f"{i+1}: {p*100:.2f}%")

