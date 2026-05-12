import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np

st.set_page_config(page_title="IA Digit Recognizer")

st.title("Reconocedor de Dígitos en Tiempo Real")
st.write("Dibuja un número del 0 al 9 en el recuadro de abajo")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("modelo_mnist.keras")

model = load_my_model()

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0

    pred = model.predict(img.reshape(1, 28, 28, 1), verbose=0)
    clase = np.argmax(pred)
    confianza = float(np.max(pred))

    st.subheader(f"Resultado: {clase}")

    if confianza < 0.7:
        st.warning(f"Confianza baja ({confianza:.2%}). ¿Podrías dibujar más claro?")
    else:
        st.success(f"Confianza alta: {confianza:.2%}")

    st.bar_chart(pred[0])
