import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import os

st.set_page_config(page_title="IA Digit Recognizer")

st.title("Reconocedor de Dígitos en Tiempo Real")
st.write("Dibuja un número del 0 al 9 en el recuadro de abajo")

# Verificar que el modelo existe
if not os.path.exists("modelo_mnist.keras"):
    st.error("No se encontró el archivo modelo_mnist.keras. Asegúrate de subirlo al repositorio.")
    st.stop()

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("modelo_mnist.keras")

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error cargando el modelo: {e}")
    st.stop()

# Botón para borrar el canvas
if st.button("🗑️ Borrar dibujo"):
    st.session_state.canvas_key = st.session_state.get("canvas_key", 0) + 1

canvas_key = st.session_state.get("canvas_key", "canvas")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=canvas_key,
)

if canvas_result.image_data is not None:
    try:
        # El canvas devuelve RGBA, necesitamos manejarlo bien
        img = canvas_result.image_data.astype('uint8')

        # Si tiene 4 canales (RGBA), convertir a BGR primero
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        img = cv2.resize(img, (28, 28))

        # Convertir a escala de grises
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img / 255.0

        # Invertir si es necesario (MNIST tiene fondo negro y dígito blanco)
        img = 1.0 - img

        pred = model.predict(img.reshape(1, 28, 28, 1), verbose=0)
        clase = np.argmax(pred)
        confianza = float(np.max(pred))

        st.subheader(f"Resultado: {clase}")

        if confianza < 0.7:
            st.warning(f"Confianza baja ({confianza:.2%}). ¿Podrías dibujar más claro?")
        else:
            st.success(f"Confianza alta: {confianza:.2%}")

        # Mostrar probabilidades
        probs = {str(i): float(pred[0][i]) for i in range(10)}
        st.bar_chart(probs)

    except Exception as e:
        st.error(f"Error procesando la imagen: {e}")
