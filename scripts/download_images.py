
import argparse
import json
from io import BytesIO
from pathlib import Path

import truststore

# Usar os certificados do sistema sem desativar a validacao HTTPS.
truststore.inject_into_ssl()

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "datasets" / "coco2017"
MANIFEST = BASE / "selected_images.json"
DESTINATION = BASE / "images" / "val2017"
IMAGE_URL = "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/{}"


def parse_args():
    """Ler as opcoes passadas ao executar o script."""
    parser = argparse.ArgumentParser(description="Baixa imagens selecionadas do COCO.")
    parser.add_argument("--limit", type=int, default=3, help="Numero maximo de imagens.")
    return parser.parse_args()


def download_image(filename):
    """Baixar uma imagem, valida-la e salvar no destino."""
    output = DESTINATION / filename

    # Evitar baixar novamente um arquivo que ja seja uma imagem valida.
    if output.exists():
        with Image.open(output) as img:
            img.verify()
        return "exists"

    response = requests.get(IMAGE_URL.format(filename), timeout=(15, 90))
    response.raise_for_status()

    # Conferir o conteudo antes de gravar o arquivo.
    with Image.open(BytesIO(response.content)) as img:
        img.verify()

    output.write_bytes(response.content)
    return "downloaded"


def main():
    """Carregar o manifesto, baixar as imagens e mostrar o resumo."""
    args = parse_args()

    # O limite controla quantas entradas do manifesto serao processadas.
    with MANIFEST.open("r", encoding="utf-8") as file:
        images = json.load(file)["images"][:args.limit]

    DESTINATION.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failed = 0

    print("=== CLASSDETECT AI ===")
    print("Iniciando download de", len(images), "imagens")

    # Uma falha individual nao impede as demais tentativas.
    for index, image in enumerate(images, start=1):
        filename = image["file_name"]

        try:
            result = download_image(filename)
            if result == "exists":
                print(f"[{index}] Ja existe: {filename}")
            else:
                print(f"[{index}] OK: {filename}")
            downloaded += 1
        except Exception as error:
            print(f"[{index}] ERRO: {filename} - {error}")
            failed += 1

    print("\n=== RESULTADO ===")
    print("Imagens disponiveis:", downloaded)
    print("Falhas:", failed)
    print("Pasta:", DESTINATION)


if __name__ == "__main__":
    main()
