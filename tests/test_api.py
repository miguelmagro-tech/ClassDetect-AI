
from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def test_detect_valid_image():

    # Criar uma imagem de teste na memoria
    image = Image.new("RGB", (100, 100), color="white")

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    # Simular a resposta do detector YOLO
    expected_response = {
        "detections": [
            {
                "class_name": "laptop",
                "confidence": 0.85,
                "bbox": {
                    "x_min": 10,
                    "y_min": 20,
                    "x_max": 80,
                    "y_max": 90
                }
            }
        ]
    }

    # Substituir temporariamente o detector real
    with patch(
        "app.main.detect_objects",
        return_value=expected_response
    ) as mock_detector:

        response = client.post(
            "/detect",
            files={
                "file": (
                    "test.png",
                    buffer.getvalue(),
                    "image/png"
                )
            }
        )

        # Verificar a resposta HTTP e o JSON
        assert response.status_code == 200
        assert response.json() == expected_response

        # Confirmar que a API chamou o detector
        mock_detector.assert_called_once()
