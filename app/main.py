from fastapi import FastAPI


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