import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import numpy as np
import os
from PIL import Image

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
    st.success("✅ Modelo cargado")
except Exception as e:
    st.error(f"❌ Error: {e}")
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

if st.button("🔮 Predecir número", type="primary"):
    if canvas_result.image_data is not None:
        try:
            # PIL procesa la imagen del canvas correctamente
            input_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            input_image_gs = input_image.convert('L')
            img_resized = input_image_gs.resize((28, 28), Image.LANCZOS)
            img_array = np.array(img_resized, dtype=np.float32)
            img_norm = img_array / 255.0
            
            # Invertir si es necesario
            if np.mean(img_norm) > 0.5:
                img_final = 1.0 - img_norm
            else:
                img_final = img_norm
            
            pred = model.predict(img_final.reshape(1, 28, 28, 1), verbose=0)
            clase = int(np.argmax(pred))
            confianza = float(np.max(pred))
            
            st.subheader(f"🎯 Número detectado: {clase}")
            
            if confianza < 0.7:
                st.warning(f"⚠️ Confianza baja: {confianza:.1%}")
            else:
                st.success(f"✅ Confianza: {confianza:.1%}")
            
            probs = {str(i): float(pred[0][i]) for i in range(10)}
            st.bar_chart(probs)
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.info("📝 Dibuja algo primero")