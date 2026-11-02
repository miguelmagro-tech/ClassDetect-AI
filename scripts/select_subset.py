
import json
import random
from pathlib import Path
from collections import defaultdict

# ==========================================
# CLASSDETECT AI - SELECAO DO DATASET COCO
# ==========================================

# Localizar automaticamente a pasta do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE = PROJECT_ROOT / "datasets" / "coco2017"
SOURCE = BASE / "annotations" / "instances_val2017.json"

# Classes escolhidas para o ClassDetect AI
CLASSES = [
    "person",
    "backpack",
    "book",
    "cell phone",
    "laptop",
    "chair",
    "bottle",
    "cup",
    "keyboard",
    "mouse",
]

# Quantidade minima de imagens por classe
QUOTA = 40
rng = random.Random(42)

# ==========================================
# 1. CARREGAR AS ANOTACOES
# ==========================================

if not SOURCE.exists():
    raise FileNotFoundError(
        f"Arquivo de anotacoes nao encontrado: {SOURCE}"
    )

with SOURCE.open("r", encoding="utf-8") as file:
    data = json.load(file)

print("Anotacoes carregadas com sucesso!")

# ==========================================
# 2. IDENTIFICAR AS CLASSES
# ==========================================

category_ids = {
    category["name"]: category["id"]
    for category in data["categories"]
    if category["name"] in CLASSES
}

if set(category_ids) != set(CLASSES):
    raise ValueError(
        "Uma ou mais classes nao foram encontradas no COCO."
    )

id_to_class = {
    category_id: name
    for name, category_id in category_ids.items()
}

# ==========================================
# 3. ORGANIZAR IMAGENS POR CLASSE
# ==========================================

images_by_class = defaultdict(set)
classes_by_image = defaultdict(set)

for annotation in data["annotations"]:
    name = id_to_class.get(annotation["category_id"])

    if name is not None:
        image_id = annotation["image_id"]

        images_by_class[name].add(image_id)
        classes_by_image[image_id].add(name)

# ==========================================
# 4. SELECIONAR AS IMAGENS
# ==========================================

# Priorizar classes menos frequentes
ordered_classes = sorted(
    CLASSES,
    key=lambda name: len(images_by_class[name])
)

selected = set()

for name in ordered_classes:
    available = sorted(images_by_class[name] - selected)
    rng.shuffle(available)

    existing = len(selected & images_by_class[name])
    missing = max(0, QUOTA - existing)

    selected.update(available[:missing])

# Verificar cobertura minima
for name in CLASSES:
    count = len(selected & images_by_class[name])

    if count < QUOTA:
        raise ValueError(
            f"Classe {name} possui apenas {count} imagens."
        )

# ==========================================
# 5. CRIAR MANIFESTO DO DATASET
# ==========================================

image_metadata = {
    image["id"]: image
    for image in data["images"]
}

manifest = {
    "source": "COCO 2017 val2017",
    "seed": 42,
    "minimum_images_per_class": QUOTA,
    "images": [
        {
            "id": image_id,
            "file_name": image_metadata[image_id]["file_name"],
            "classes": sorted(classes_by_image[image_id])
        }
        for image_id in sorted(selected)
    ]
}

# ==========================================
# 6. SALVAR RESULTADOS
# ==========================================

OUTPUT = PROJECT_ROOT / "runs" / "dataset_preview" / "selected_images.json"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as file:
    json.dump(
        manifest,
        file,
        indent=2,
        ensure_ascii=False
    )

# ==========================================
# 7. MOSTRAR RESULTADOS
# ==========================================

print("\n========== CLASSDETECT AI ==========")
print("Dataset: COCO 2017")
print("Total de imagens selecionadas:", len(selected))

print("\nDistribuicao das classes:")

for name in CLASSES:
    count = len(selected & images_by_class[name])
    print(f"{name}: {count} imagens")

print("\nManifesto criado com sucesso!")
print("Arquivo:", OUTPUT)

print("\nPreparacao inicial concluida!")
