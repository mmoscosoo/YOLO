import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

st.set_page_config(
    page_title="VisionCam AI",
    page_icon="📸",
    layout="wide"
)

# Estilos personalizados
st.markdown("""
    <style>
    html, body, .main, .stApp, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
        background-color: #2E4053 !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #F0F3F4 !important;
    }
    html, body, h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #FDFEFE !important;
        text-align: left !important;
    }
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-16txtl3 {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path='yolov5s.pt'):
    try:
        import yolov5
        try:
            return yolov5.load(path, weights_only=False)
        except TypeError:
            return yolov5.load(path)
    except Exception:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        except Exception as e:
            st.error(f"❌ No se pudo cargar el modelo: {str(e)}")
            return None

# Encabezado
st.title("📸 VisionCam AI")
st.markdown("Explora el mundo que te rodea con inteligencia artificial. Esta app detecta objetos en tiempo real usando YOLOv5 y tu cámara.")

with st.spinner("Inicializando modelo..."):
    model = load_model()

if model:
    st.sidebar.header("🎛️ Ajustes de Detección")

    model.conf = st.sidebar.slider('Nivel de confianza', 0.1, 1.0, 0.35, 0.01)
    model.iou = st.sidebar.slider('Umbral de IoU', 0.1, 1.0, 0.45, 0.01)
    st.sidebar.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")

    st.sidebar.subheader("🔬 Avanzado")
    model.agnostic = st.sidebar.checkbox("Clasificación agnóstica", False)
    model.multi_label = st.sidebar.checkbox("Etiquetas múltiples", False)
    model.max_det = st.sidebar.number_input("Máx. objetos a detectar", 100, 3000, 1000, 50)

    img_input = st.camera_input("Toma una foto para analizar")

    if img_input:
        img_data = img_input.getvalue()
        img_cv2 = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("Analizando imagen..."):
            try:
                result = model(img_cv2)
            except Exception as e:
                st.error(f"Error en la detección: {e}")
                st.stop()

        predictions = result.pred[0]
        boxes, scores, categories = predictions[:, :4], predictions[:, 4], predictions[:, 5]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Resultados")
            labels = model.names
            counts = {}

            for cat in categories:
                idx = int(cat.item()) if hasattr(cat, 'item') else int(cat)
                counts[idx] = counts.get(idx, 0) + 1

            rows = []
            for cat_id, count in counts.items():
                label = labels[cat_id]
                conf = scores[categories == cat_id].mean().item() if len(scores) > 0 else 0
                rows.append({
                    "Objeto": label,
                    "Veces detectado": count,
                    "Confianza prom.": f"{conf:.2f}"
                })

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("Objeto")["Veces detectado"])
            else:
                st.info("No se detectaron objetos con los parámetros actuales.")
                st.caption("Ajusta la confianza o el IoU en la barra lateral.")

        with col2:
            st.subheader("🖼️ Imagen procesada")
            result.render()
            st.image(img_cv2, channels="BGR", use_container_width=True)
else:
    st.error("Fallo al cargar el modelo. Verifica tus dependencias e intenta nuevamente.")
    st.stop()

st.markdown("---")
st.caption("📌 Aplicación desarrollada con Streamlit + YOLOv5 | Proyecto educativo")

try:
    st.image("visioncam_banner.png", use_container_width=True)
except:
    st.warning("No se pudo cargar la imagen 'visioncam_banner.png'.")
