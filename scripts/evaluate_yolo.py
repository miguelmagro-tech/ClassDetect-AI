#-----------------------------------
#IMPOTANDO BIBLIOTECAS
#-----------------------------------



import argparse
import json
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO





ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "datasets" / "coco2017"


MANIFEST = BASE / "selected_images.json"
ANNOTATIONS = BASE / "annotations" / "instances_val2017.json"
IMAGES_DIR = BASE / "images" / "val2017"
OUTPUT_DIR = ROOT / "runs" / "evaluation"


TARGET_CLASSES = {
    "person", "backpack", "book", "cell phone",
    "laptop", "chair", "bottle", "cup",
    "keyboard", "mouse"
}


def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )
    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    with MANIFEST.open(encoding="utf-8") as file:
        manifest = json.load(file)

    with ANNOTATIONS.open(encoding="utf-8") as file:
        coco = json.load(file)

    if args.limit < 1:
        parser.error("--limit deve ser maior que zero")

    images = manifest["images"][:args.limit]
    image_ids = {image["id"] for image in images}

    category_names = {
        item["id"]: item["name"]
        for item in coco["categories"]
    }

    # Organizar anotacoes reais por imagem
    annotations_by_image = defaultdict(list)

    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]

        if image_id not in image_ids:
            continue

        # Esta avaliacao simples nao considera crowd
        if annotation.get("iscrowd", 0):
            continue

        class_name = category_names[annotation["category_id"]]

        if class_name not in TARGET_CLASSES:
            continue

        x, y, width, height = annotation["bbox"]

        annotations_by_image[image_id].append({
            "class_name": class_name,
            "bbox": [x, y, x + width, y + height]
        })

    print("\n=== CLASSDETECT AI ===")
    print("Iniciando validacao do YOLO11n")
    print("Imagens selecionadas:", len(images))

    model = YOLO("yolo11n.pt")

    class_ids = [
        class_id
        for class_id, name in model.names.items()
        if name in TARGET_CLASSES
    ]

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for index, image in enumerate(images, start=1):
        image_path = IMAGES_DIR / image["file_name"]

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        results = model.predict(
            source=str(image_path),
            classes=class_ids,
            conf=0.20,
            imgsz=960,
            device="cpu",
            verbose=False
        )

        ground_truth = annotations_by_image[image["id"]]
        unmatched = set(range(len(ground_truth)))

        tp = 0
        fp = 0

        # Comparar cada deteccao com a anotacao real
        for box in results[0].boxes:
            class_name = model.names[int(box.cls.item())]
            predicted_bbox = box.xyxy[0].tolist()

            best_iou = 0.0
            best_match = None

            for gt_index in unmatched:
                gt = ground_truth[gt_index]

                if gt["class_name"] != class_name:
                    continue

                iou = calculate_iou(
                    predicted_bbox,
                    gt["bbox"]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_match = gt_index

            if best_match is not None and best_iou >= 0.50:
                tp += 1
                unmatched.remove(best_match)
            else:
                fp += 1

        fn = len(unmatched)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(
            f"[{index}/{len(images)}] "
            f"{image['file_name']} | "
            f"TP={tp} FP={fp} FN={fn}"
        )

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0 else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0 else 0.0
    )

    summary = {
        "images_evaluated": len(images),
        "confidence_threshold": 0.20,
        "iou_threshold": 0.50,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "image_size": 960,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "summary.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n=== RESULTADO FINAL ===")
    print(json.dumps(summary, indent=4))
    print("\nRelatorio salvo em:", output_path)


if __name__ == "__main__":
    main()