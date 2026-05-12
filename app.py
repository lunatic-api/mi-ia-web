import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="IA Digit Recognizer")

st.title("Reconocedor de Dígitos en Tiempo Real")
st.write("Sube una imagen de un número del 0 al 9")

if not os.path.exists("modelo_mnist.keras"):
    st.error("No se encontró modelo_mnist.keras")
    st.stop()

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("modelo_mnist.keras")

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error cargando modelo: {e}")
    st.stop()

uploaded = st.file_uploader("Elige una imagen (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("L")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Imagen original", width=200)
    
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array
    
    with col2:
        st.image(img_array, caption="Procesada (28x28)", width=200)
    
    pred = model.predict(img_array.reshape(1, 28, 28, 1), verbose=0)
    clase = int(np.argmax(pred))
    confianza = float(np.max(pred))
    
    st.subheader(f"Número detectado: {clase}")
    
    if confianza < 0.7:
        st.warning(f"Confianza baja ({confianza:.1%}). Prueba con otra imagen más clara.")
    else:
        st.success(f"Confianza: {confianza:.1%}")
    
    probs = {str(i): float(pred[0][i]) for i in range(10)}
    st.bar_chart(probs)