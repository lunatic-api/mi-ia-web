import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import os

st.set_page_config(page_title="IA Digit Recognizer")

st.title("Reconocedor de Dígitos en Tiempo Real")
st.write("Dibuja un número del 0 al 9 en el recuadro negro y pulsa 'Predecir'")

if not os.path.exists("modelo_mnist.keras"):
    st.error("No se encontró modelo_mnist.keras")
    st.stop()

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("modelo_mnist.keras")

try:
    model = load_my_model()
    st.success("Modelo cargado")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",
    stroke_width=20,
    stroke_color="rgba(255, 255, 255, 1)",
    background_color="rgba(0, 0, 0, 1)",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
    update_streamlit=False,
)

if st.button("Predecir número"):
    if canvas_result.image_data is not None:
        try:
            img = canvas_result.image_data.astype('uint8')
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            else:
                img_gray = img
            img_resized = cv2.resize(img_gray, (28, 28))
            img_norm = img_resized.astype('float32') / 255.0
            img_final = 1.0 - img_norm
            pred = model.predict(img_final.reshape(1, 28, 28, 1), verbose=0)
            clase = int(np.argmax(pred))
            confianza = float(np.max(pred))
            st.subheader(f"Número: {clase}")
            if confianza < 0.7:
                st.warning(f"Confianza baja: {confianza:.1%}")
            else:
                st.success(f"Confianza: {confianza:.1%}")
            probs = {str(i): float(pred[0][i]) for i in range(10)}
            st.bar_chart(probs)
        except Exception as e:
            st.error(f"Error: {e}")