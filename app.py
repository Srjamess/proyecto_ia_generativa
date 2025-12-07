import streamlit as st
from diffusers import DiffusionPipeline
import torch
from datetime import datetime
import os
import base64

# ----------------------------------------------------
# 1. CONFIGURACIÓN GENERAL Y CARGA DEL MODELO
# ----------------------------------------------------
st.set_page_config(
    page_title="Generador Educativo IA",
    page_icon="🎨",
    layout="wide"
)

@st.cache_resource
def load_model():
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")
    return pipe

pipe = load_model()


# ----------------------------------------------------
# 2. FUNCIONES AUXILIARES
# ----------------------------------------------------
def agregar_estilo(prompt, estilo):
    estilos = {
        "Realista": "realistic, detailed, high resolution",
        "Ilustración educativa": "flat illustration, colorful, infographic style",
        "Acuarela": "watercolor, soft edges, artistic",
        "Minimalista": "flat minimalism, clean lines, soft palette"
    }
    
    return f"{prompt}, {estilos[estilo]}"


def guardar_imagen(imagen):
    if not os.path.exists("resultados"):
        os.makedirs("resultados")
    filename = f"resultados/img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    imagen.save(filename)
    return filename


def download_button(file_path):
    with open(file_path, "rb") as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="imagen.png">📥 Descargar imagen</a>'
    st.markdown(href, unsafe_allow_html=True)


# ----------------------------------------------------
# 3. SIDEBAR - CONFIGURACIONES AVANZADAS
# ----------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    estilo = st.selectbox(
        "Estilo visual",
        ["Ilustración educativa", "Realista", "Acuarela", "Minimalista"]
    )

    steps = st.slider("Pasos de inferencia", 10, 50, 20)
    guidance = st.slider("Guidance scale", 5.0, 15.0, 7.5)

    seed = st.number_input("Seed (opcional)", value=0, min_value=0, help="Para reproducir resultados")

    st.markdown("---")
    st.header("📊 Evaluación de imagen")

    claridad = st.slider("Claridad", 1, 5, 3)
    relevancia = st.slider("Relevancia con el prompt", 1, 5, 3)
    estetica = st.slider("Estética general", 1, 5, 3)

    evaluar = st.button("Guardar evaluación")


# ----------------------------------------------------
# 4. INTERFAZ PRINCIPAL
# ----------------------------------------------------
st.title(" Generador de Imágenes Educativas con IA")

prompt = st.text_input(
    "Ingresa un tema educativo:",
    ""
)

generar = st.button("✨ Generar Imagen")

# ----------------------------------------------------
# 5. GENERACIÓN DE IMAGEN
# ----------------------------------------------------
if generar:
    st.subheader("🖼 Imagen generada:")

    prompt_final = agregar_estilo(prompt, estilo)

    with st.spinner("Generando imagen..."):
        if seed != 0:
            generator = torch.manual_seed(seed)
        else:
            generator = None

        raw_image = pipe(
            prompt_final,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]

        filename = guardar_imagen(raw_image)

        st.image(raw_image, use_column_width=True)
        download_button(filename)

        st.success(f"Imagen guardada en: {filename}")


# ----------------------------------------------------
# 6. EVALUACIÓN - SE GUARDA EN CSV PARA EL INFORME
# ----------------------------------------------------
if evaluar:
    import csv

    if not os.path.exists("evaluaciones.csv"):
        with open("evaluaciones.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt", "claridad", "relevancia", "estetica"])

    with open("evaluaciones.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([prompt, claridad, relevancia, estetica])

    st.success("Evaluación guardada correctamente.")


# ----------------------------------------------------
# 7. PORTAFOLIO DE IMÁGENES
# ----------------------------------------------------
st.markdown("---")
st.header("🖼 Portafolio de imágenes generadas")

if os.path.exists("resultados"):
    imgs = os.listdir("resultados")
    cols = st.columns(4)

    for i, img in enumerate(imgs):
        with cols[i % 4]:
            st.image(f"resultados/{img}", caption=img, use_column_width=True)
else:
    st.info("Aún no has generado imágenes.")
