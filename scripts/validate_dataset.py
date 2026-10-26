
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image

# ==========================================
# CLASSDETECT AI - VALIDACAO DO DATASET
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "datasets" / "coco2017"

MANIFEST = BASE / "selected_images.json"
ANNOTATIONS = BASE / "annotations" / "instances_val2017.json"
IMAGES_DIR = BASE / "images" / "val2017"

# Carregar arquivos JSON
with MANIFEST.open(encoding="utf-8") as file:
    manifest = json.load(file)

with ANNOTATIONS.open(encoding="utf-8") as file:
    coco = json.load(file)

# Metadados originais
image_info = {
    image["id"]: image
    for image in coco["images"]
}

category_names = {
    category["id"]: category["name"]
    for category in coco["categories"]
}

# Classes anotadas por imagem
classes_by_image = defaultdict(set)

for annotation in coco["annotations"]:
    classes_by_image[annotation["image_id"]].add(
        category_names[annotation["category_id"]]
    )

selected = manifest["images"]

valid = 0
errors = []
class_counts = defaultdict(int)
seen_ids = set()

print("\n=== CLASSDETECT AI ===")
print("Validando imagens e anotacoes...\n")

for item in selected:
    image_id = item["id"]
    filename = item["file_name"]
    path = IMAGES_DIR / filename

    if image_id in seen_ids:
        errors.append(f"ID duplicado: {image_id}")
        continue

    seen_ids.add(image_id)

    if image_id not in image_info:
        errors.append(f"ID ausente no COCO: {image_id}")
        continue

    metadata = image_info[image_id]

    if filename != metadata["file_name"]:
        errors.append(f"Nome incorreto: {filename}")
        continue

    if not path.exists():
        errors.append(f"Imagem ausente: {filename}")
        continue

    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()

        if (
            width != metadata["width"]
            or height != metadata["height"]
        ):
            errors.append(f"Dimensoes diferentes: {filename}")
            continue

        expected_classes = set(item["classes"])
        original_classes = classes_by_image[image_id]

        if not expected_classes.issubset(original_classes):
            errors.append(f"Classes inconsistentes: {filename}")
            continue

        valid += 1

        for name in expected_classes:
            class_counts[name] += 1

    except Exception as error:
        errors.append(f"{filename}: {error}")

print("Imagens no manifesto:", len(selected))
print("Imagens validadas:", valid)
print("Problemas encontrados:", len(errors))

print("\n=== COBERTURA DAS CLASSES ===")

all_classes = [
    "person", "backpack", "book",
    "cell phone", "laptop", "chair",
    "bottle", "cup", "keyboard", "mouse"
]

for name in all_classes:
    print(f"{name}: {class_counts[name]} imagens")

    if class_counts[name] < 40:
        errors.append(f"Cobertura insuficiente: {name}")

print("\n=== RESULTADO FINAL ===")

if errors:
    print("VALIDACAO REPROVADA")
    for error in errors[:15]:
        print("ERRO:", error)
    raise SystemExit(1)
else:
    print("DATASET VALIDADO COM SUCESSO!")
    print("Todas as imagens e classes verificadas.")
