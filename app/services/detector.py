
from functools import lru_cache
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


# Raiz do projeto ClassDetect AI
ROOT = Path(__file__).resolve().parents[2]

# Configuracao definida na TASK-007
CONFIDENCE = 0.20
IMAGE_SIZE = 960

# IDs das 10 classes do COCO utilizadas no projeto
CLASS_IDS = [0, 24, 39, 41, 56, 63, 64, 66, 67, 73]


@lru_cache(maxsize=1)
def get_model():
    """Carrega o YOLO11n apenas uma vez por processo."""

    model_path = ROOT / "yolo11n.pt"

    if model_path.exists():
        return YOLO(str(model_path))

    return YOLO("yolo11n.pt")


def detect_objects(image: Image.Image) -> dict:
    """Recebe uma imagem e retorna as deteccoes."""

    model = get_model()

    results = model.predict(
        source=image,
        classes=CLASS_IDS,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False
    )

    detections = []

    for box in results[0].boxes:
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

    return {"detections": detections}
