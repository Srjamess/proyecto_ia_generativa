import torch
from safetensors.torch import load_file
import os

# -------------------------------------------------------
# CONFIGURACIÓN - RUTAS DEL LORA
# -------------------------------------------------------

# Archivo LoRA creado por Kohya SS
kohya_lora = r"E:\lora_trainer\Kohya_ss-GUI-LoRA-Portable-main\Kohya_ss-GUI-LoRA-Portable-main\dataset\outputs\last.safetensors"

# Archivo LoRA convertido (salida final)
output_path = r"E:\lora_trainer\output_lora\converted_lora_diffusers.bin"

print("------------------------------------------------------------")
print(" CONVERSIÓN DE LORA KOHYA  →  FORMATO DIFFUSERS")
print("------------------------------------------------------------")

# Validar archivo de entrada
if not os.path.exists(kohya_lora):
    print("❌ ERROR: No se encuentra el archivo LoRA de Kohya:")
    print(kohya_lora)
    exit()

print("✔ Archivo LoRA encontrado.")
print("Cargando archivo .safetensors...")

# Cargar archivo safetensors
state = load_file(kohya_lora)

print("✔ Archivo cargado con éxito.")
print("Convirtiendo claves LoRA...")

# Extraer solo las claves que diffusers puede usar
converted = {}

for key, value in state.items():
    if "lora" in key or "alpha" in key:
        converted[key] = value

print(f"✔ Total de capas LoRA convertidas: {len(converted)}")

# Crear carpeta si no existe
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print("Guardando archivo convertido...")

torch.save(converted, output_path)

print("------------------------------------------------------------")
print(" ✔ CONVERSIÓN COMPLETADA")
print(f" ✔ Archivo guardado en: {output_path}")
print("------------------------------------------------------------")
