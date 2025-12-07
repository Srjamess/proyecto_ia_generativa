from diffusers import DiffusionPipeline
import torch

def main():
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")

    prompt = "gatos jugando ajedrez en un parque soleado, estilo pintura impresionista"
    image = pipe(prompt).images[0]
    image.save("resultados/sistema_solar_prueba.png")

    print("Imagen generada y guardada en resultados/sistema_solar_prueba.png")

if __name__ == "__main__":
    main()
