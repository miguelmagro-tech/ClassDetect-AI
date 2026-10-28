
import json
from pathlib import Path
from ultralytics import YOLO
import argparse

# ==========================================
# CLASSDETECT AI - PRIMEIRO TESTE YOLO11n
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

BASE = ROOT / "datasets" / "coco2017"
MANIFEST = BASE / "selected_images.json"
IMAGES_DIR = BASE / "images" / "val2017"

# IDs das 10 classes no modelo YOLO COCO
CLASS_IDS = [0, 24, 39, 41, 56, 63, 64, 66, 67, 73]

# Carregar a lista de imagens selecionadas
with MANIFEST.open("r", encoding="utf-8") as file:
    manifest = json.load(file)

# Escolher a primeira imagem do dataset

# Selecionar uma imagem anotada com a classe "bottle" (ID 39) para o teste

# Permitir escolher a classe pelo terminal
parser = argparse.ArgumentParser()

parser.add_argument(
    "--classe",
    default="bottle",
    choices=[
        "person", "backpack", "book", "cell phone",
        "laptop", "chair", "bottle", "cup",
        "keyboard", "mouse"
    ]
)

args = parser.parse_args()

# Selecionar uma imagem com a classe solicitada
image_name = next(
    item["file_name"]
    for item in manifest["images"]
    if args.classe in item["classes"]
)


image_path = IMAGES_DIR / image_name

if not image_path.exists():
    raise FileNotFoundError(
        f"Imagem nao encontrada: {image_path}"
    )

print("\n=== CLASSDETECT AI ===")
print("Imagem selecionada:", image_name)

# Carregar o modelo YOLO11n pre-treinado
print("\nCarregando modelo YOLO11n...")
model = YOLO("yolo11n.pt")

# Executar a deteccao
print("Executando deteccao de objetos...")

results = model.predict(
    source=str(image_path),
    classes=CLASS_IDS,  # Filtrar apenas a classe "bottle"
    conf=0.25,
    device="cpu",
    verbose=False
)

result = results[0]


# Organizar as deteccoes para a futura API
detections = []

for box in result.boxes:
    class_id = int(box.cls.item())
    class_name = model.names[class_id]
    confidence = float(box.conf.item())

    x_min, y_min, x_max, y_max = [
        int(value)
        for value in box.xyxy[0].tolist()
    ]

    detections.append({
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "bbox": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max
        }
    })

response = {"detections": detections}

print("\n=== RESULTADO JSON ===")
print(json.dumps(response, indent=4, ensure_ascii=False))

print("\nTeste finalizado!")



# Salvar imagem com os objetos detectados
OUTPUT_DIR = ROOT / "runs" / "test_yolo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_path = OUTPUT_DIR / image_name

result.save(filename=str(output_path))

print("\nImagem com deteccoes salva em:")
print(output_path)
