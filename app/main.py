
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

from app.services.detector import detect_objects



#criar a aplicação do FastAPI
app = FastAPI(
    title="ClassDetect AI",
    description="API para detecção de classes em imagens usando YOLO11n",
    version="0.1.0",

)


#rota inicial para verificar o funcionamento da APi

@app.get("/")
def home():
    return {
        "message": "Bem-vindo à API ClassDetect AI!",
        "status": "API está funcionando corretamente."
            
            }


# Endpoint para receber uma imagem e detectar objetos
@app.post("/detect")
def detect(file: UploadFile = File(...)):

    # Verificar se o arquivo enviado e uma imagem
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="O arquivo enviado deve ser uma imagem."
        )

    # Abrir e converter a imagem para RGB
    try:
        with Image.open(file.file) as image:
            image_rgb = image.convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Imagem invalida ou corrompida."
        )

    # Executar o YOLO11n e devolver o resultado
    return detect_objects(image_rgb)
