
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




def test_detect_invalid_file():

    # Simular o envio de um arquivo de texto
    response = client.post(
        "/detect",
        files={
            "file": (
                "teste.txt",
                b"Este arquivo nao e uma imagem.",
                "text/plain"
            )
        }
    )

    # A API deve rejeitar o arquivo
    assert response.status_code == 400

    # Conferir a mensagem de erro
    assert response.json() == {
        "detail": "O arquivo enviado deve ser uma imagem."
    }




def test_detect_without_file():

    # Enviar uma requisicao sem arquivo
    response = client.post("/detect")

    # O FastAPI deve rejeitar a requisicao
    assert response.status_code == 422

    # Conferir se o erro aponta para o arquivo ausente
    errors = response.json()["detail"]

    assert any(
        "file" in error["loc"]
        for error in errors
    )




def test_detect_corrupted_image():

    # Simular um arquivo PNG com conteudo invalido
    response = client.post(
        "/detect",
        files={
            "file": (
                "imagem.png",
                b"Este conteudo nao e uma imagem valida.",
                "image/png"
            )
        }
    )

    # A API deve identificar a imagem corrompida
    assert response.status_code == 400

    # Verificar a mensagem retornada
    assert response.json() == {
        "detail": "Imagem invalida ou corrompida."
    }


    
def test_detect_real_yolo():
    from pathlib import Path
    import pytest

    # Localizar uma imagem real do nosso dataset
    root = Path(__file__).resolve().parent.parent

    image_path = (
        root
        / "datasets"
        / "coco2017"
        / "images"
        / "val2017"
        / "000000001675.jpg"
    )

    if not image_path.exists():
        pytest.skip("Imagem COCO nao disponivel localmente")

    # Enviar a imagem para a API sem simular o YOLO
    with image_path.open("rb") as image_file:
        response = client.post(
            "/detect",
            files={
                "file": (
                    image_path.name,
                    image_file,
                    "image/jpeg"
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    # Validar a estrutura da resposta
    assert "detections" in data
    assert isinstance(data["detections"], list)

    # Esta imagem possui um notebook que ja detectamos
    assert any(
        obj["class_name"] == "laptop"
        for obj in data["detections"]
    )

    # Conferir os campos de cada deteccao
    for obj in data["detections"]:
        assert "class_name" in obj
        assert 0 <= obj["confidence"] <= 1

        bbox = obj["bbox"]

        assert set(bbox) == {
            "x_min", "y_min", "x_max", "y_max"
        }

