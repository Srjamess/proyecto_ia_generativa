# Generador de Imágenes Educativas con IA  
## Stable Diffusion 1.5 + LoRA + Streamlit  
![Banner](https://dummyimage.com/1200x300/4c6ef5/ffffff&text=Generador+de+Imágenes+Educativas+con+IA)

---

<p align="center">
  <img src="https://img.shields.io/badge/IA-Generativa-blueviolet?style=for-the-badge">
  <img src="https://img.shields.io/badge/StableDiffusion-1.5-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/LoRA-Train-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-App-green?style=for-the-badge">
</p>

---

## 📘 Descripción General

Este proyecto implementa una solución de **Inteligencia Artificial Generativa** capaz de producir **imágenes educativas** a partir de texto, utilizando:

- **Stable Diffusion 1.5** como modelo base  
- **LoRA personalizado** entrenado con ilustraciones educativas  
- Una aplicación **Streamlit** para generación, análisis y portafolio visual  

El sistema está diseñado para docentes, estudiantes y creadores que deseen generar material gráfico educativo de manera rápida, coherente y personalizada.

---

# 📚 Tabla de Contenidos
1. [Estructura del Proyecto](#estructura-del-proyecto)
2. [Objetivo del Proyecto](#objetivo-del-proyecto)
3. [Tecnologías Utilizadas](#tecnologías-utilizadas)
4. [Arquitectura General](#arquitectura-general)
5. [Dataset Utilizado](#dataset-utilizado)
6. [Entrenamiento del LoRA](#entrenamiento-del-lora)
7. [Conversión del LoRA a Diffusers](#conversión-del-lora-a-diffusers)
8. [Aplicación Streamlit](#aplicación-streamlit)
9. [Experimentos Realizados](#experimentos-realizados)
10. [Reflexión Ética y Sesgos](#reflexión-ética-y-sesgos)
11. [Impacto Educativo](#impacto-educativo)
12. [Propuestas de Mejora](#propuestas-de-mejora)
13. [Requerimientos Técnicos](#requerimientos-técnicos)
14. [Cómo Ejecutar el Proyecto](#cómo-ejecutar-el-proyecto)
15. [Conclusiones](#conclusiones)
16. [Créditos](#créditos)

---

# 🗂️ Estructura del Proyecto

```plaintext
proyecto_ia_generativa/
│── app.py                     # Aplicación Streamlit
│── convert_lora.py            # Conversión Kohya → Diffusers
│── resultados/                # Imágenes generadas
│── evaluaciones.csv           # Evaluaciones
│── venv/                      # Entorno virtual
└── README.md                  # Documentación del proyecto
```

---

# 🎯 Objetivo del Proyecto

Este proyecto permite:

- Generar imágenes educativas a partir de *prompts* en lenguaje natural.  
- Aplicar un estilo visual uniforme mediante LoRA.  
- Evaluar imágenes mediante criterios pedagógicos.  
- Construir un **portafolio visual educativo** reutilizable.

Su propósito es apoyar procesos educativos mediante contenido visual accesible y personalizable.

---

# 🛠️ Tecnologías Utilizadas

### **Modelos y librerías**
- Stable Diffusion 1.5  
- LoRA Training  
- Diffusers (Hugging Face)  
- Transformers  
- PyTorch  

### **Aplicación Web**
- Streamlit  
- Pillow  
- CSV para registro de evaluaciones  

### **Entrenamiento LoRA**
- Kohya SS LoRA Trainer  
- Método DreamBooth / train_network  
- Rank 4 · Resolución 512x512 · AdamW 8-bit  

---

# 🧱 Arquitectura General

```
Dataset educativo  
     ↓  
Entrenamiento LoRA (Kohya SS)  
     ↓  
Modelo LoRA (.safetensors)  
     ↓  
Conversión a Diffusers  
     ↓  
App Streamlit  
     ↓  
Generación + Evaluación + Portafolio
```

---

# 🖼️ Dataset Utilizado

El dataset consiste en ilustraciones estilo:

- Infografía  
- Flat design  
- Colores suaves  

Estructura:

```plaintext
dataset/
    1_educativo/
        imagen_01.png
        imagen_02.jpg
```

> La carpeta debe iniciar con número + guion (`1_nombre`), requerido por Kohya.

---

# 🔧 Entrenamiento del LoRA

Parámetros clave:

- Rank: 4  
- LR: 1e-4  
- Optimizer: AdamW 8-bit  
- Resolución: 512×512  
- Batch size: 1  

### ❌ ¿Por qué NO se usó Google Colab?

- Sesiones se cierran inesperadamente  
- Incompatibilidades con Diffusers / Transformers  
- Problemas con funciones eliminadas (`cached_download`)  
- Falta de persistencia  
- Requerimientos de VRAM altos  

➡️ Se entrenó localmente con Kohya, logrando estabilidad y control total.

---

# 🔄 Conversión del LoRA a Diffusers

Kohya produce archivos `.safetensors` no compatibles directamente con Diffusers.

Se usa `convert_lora.py` para obtener:

```
converted_lora_diffusers.bin
```

Este archivo se inyecta dentro del UNet del pipeline de Stable Diffusion.

---

# 🌐 Aplicación Streamlit

La app permite:

- Ingresar prompt  
- Elegir estilo (educativo, minimalista, realista, acuarela)  
- Ajustar hiperparámetros  
- Generar imagen  
- Guardarla automáticamente  
- Evaluarla según claridad, estética y relevancia  
- Ver un **portafolio visual** de todas las imágenes generadas  

---

# 🧪 Experimentos Realizados

### **1. Con LoRA vs Sin LoRA**
| Sin LoRA | Con LoRA |
|----------|----------|
| Resultado genérico | Estilo educativo claro |
| Menos coherencia | Mejor composición |
| Más ruido visual | Colores planos y didácticos |

**Conclusión:** El LoRA mejora significativamente el estilo educativo.

---

### **2. Variación del Guidance Scale**
| Valor | Resultado |
|-------|-----------|
| 5.0   | Más creativo, menos preciso |
| 7.5–10 | Balance ideal |
| 12    | Excesivamente literal |

---

### **3. Variación del número de steps**
| Steps | Resultado |
|--------|-----------|
| 10     | Imagen borrosa |
| 30     | Calidad óptima |
| 50     | Alto detalle, lento |

---

# ⚖️ Reflexión Ética y Sesgos

### Posibles sesgos:
- Falta de diversidad cultural  
- Estilo dependiente del dataset  
- Sobre-representación de ciertos colores o formas  

### Mitigación:
- Dataset más diverso  
- Supervisión docente  
- Prompts explícitos sobre inclusión  

### Riesgos:
- Desinformación visual  
- Uso inapropiado del contenido  
- Derechos de autor  

---

# 🎓 Impacto Educativo

Beneficios:

- Creación rápida de ilustraciones educativas  
- Material visual personalizado  
- Apoyo a docentes con poca experiencia en diseño  

Requiere:

- Validación humana  
- Uso responsable  

---

# 🚀 Propuestas de Mejora

- Integración con **ControlNet**  
- Múltiples LoRAs por área (biología, historia, infantil, etc.)  
- Generador automático de prompts educativos  
- Validador semántico del contenido  
- Ejecución en GPU dentro de Streamlit  

---

# 📦 Requerimientos Técnicos

```
diffusers==0.24.0
transformers==4.30.2
huggingface_hub==0.16.4
accelerate==0.20.3
safetensors==0.3.2
torch
streamlit
pillow
tqdm
```

---

# ▶️ Cómo Ejecutar el Proyecto

### **1. Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux
.env\Scriptsctivate   # Windows
```

### **2. Instalar dependencias**
```bash
pip install -r requirements.txt
```

### **3. Convertir LoRA**
```bash
python convert_lora.py
```

### **4. Ejecutar aplicación**
```bash
streamlit run app.py
```

Ir a:

```
http://localhost:8501/
```

---

# 🏁 Conclusiones

- Stable Diffusion + LoRA pueden adaptarse exitosamente al ámbito educativo.  
- La app Streamlit integra entrenamiento, conversión e inferencia de forma simple.  
- El sistema **no reemplaza al docente**, sino que potencia su creatividad visual.  
- Es necesario abordar temas éticos y garantizar la calidad del contenido generado.  

---

# 👤 Créditos

Proyecto creado por **James Sánchez, Patricia Franco**  
Asistencia técnica generada con IA.

---

