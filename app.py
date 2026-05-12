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
            img_rgba = canvas_result.image_data.astype('uint8')
            
            # Crear imagen negra base
            img_mnist = np.zeros((280, 280), dtype=np.float32)
            
            # Copiar solo píxeles blancos del trazo
            for y in range(280):
                for x in range(280):
                    if img_rgba[y, x, 0] > 100 or img_rgba[y, x, 1] > 100 or img_rgba[y, x, 2] > 100:
                        img_mnist[y, x] = 1.0
            
            # Redimensionar a 28x28
            img_pil = Image.fromarray((img_mnist * 255).astype('uint8'), mode='L')
            img_28 = img_pil.resize((28, 28), Image.LANCZOS)
            img_final = np.array(img_28, dtype=np.float32) / 255.0
            
            # Mostrar imagen procesada para verificar
            st.write("**Imagen procesada (28x28):**")
            st.image(img_final, width=150)
            
            # Predicción
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