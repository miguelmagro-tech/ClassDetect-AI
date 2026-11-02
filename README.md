
# ClassDetect AI

**Projeto Final — SC-TEC | Módulo 2**

Sistema de detecção de objetos em imagens utilizando Inteligência Artificial, Visão Computacional, YOLO11n e FastAPI.

## 1. Sobre o projeto

O ClassDetect AI é um projeto de Visão Computacional desenvolvido para identificar objetos presentes em imagens de ambientes de estudo, como salas de aula, bibliotecas e espaços de trabalho.

O sistema utiliza o modelo YOLO11n, disponibilizado pela biblioteca Ultralytics, para realizar a detecção de objetos.

A aplicação também disponibiliza uma API REST desenvolvida com FastAPI, permitindo enviar imagens e receber os resultados das detecções em formato JSON.

Cada detecção contém:
- Nome da classe identificada;
- Nível de confiança da previsão;
- Coordenadas da caixa delimitadora (bounding box).

## 2. Objetivos

### Objetivo geral

Desenvolver uma aplicação de Inteligência Artificial capaz de identificar objetos em imagens e disponibilizar os resultados por meio de uma API REST.

### Objetivos específicos

- Utilizar o dataset COCO 2017 como referência.
- Selecionar 10 classes relacionadas a ambientes de estudo.
- Integrar o modelo pré-treinado YOLO11n.
- Avaliar os resultados de detecção.
- Desenvolver uma API utilizando FastAPI.
- Implementar testes automatizados.
- Organizar o desenvolvimento com Git, GitHub e Trello.

## 3. Dataset e classes utilizadas

O projeto utiliza imagens do dataset COCO 2017, um conjunto de dados amplamente utilizado em tarefas de detecção de objetos.

Foi preparado um subconjunto de imagens para os testes e a validação do sistema.

O modelo trabalha com as seguintes 10 classes:

| Classe | Nome no YOLO |
|--------|--------------|
| Pessoa | person |
| Mochila | backpack |
| Livro | book |
| Celular | cell phone |
| Notebook | laptop |
| Cadeira | chair |
| Garrafa | bottle |
| Copo | cup |
| Teclado | keyboard |
| Mouse | mouse |

O projeto utiliza pesos pré-treinados do YOLO11n, sem realizar treinamento próprio do modelo.

## 4. Tecnologias utilizadas

| Tecnologia | Finalidade |
|------------|------------|
| Python 3.12 | Linguagem principal |
| YOLO11n | Modelo de detecção de objetos |
| Ultralytics | Carregamento e inferência do YOLO |
| FastAPI | Desenvolvimento da API REST |
| Uvicorn | Execução do servidor da API |
| Pillow | Leitura e processamento de imagens |
| Pytest | Execução de testes automatizados |
| HTTPX | Suporte ao cliente de testes da API |
| Git | Controle de versões |
| GitHub | Hospedagem e integração do código |
| Trello | Gerenciamento das atividades Kanban |

## 5. Funcionamento do sistema

O funcionamento da aplicação segue cinco etapas:

1. O usuário envia uma imagem para o endpoint `POST /detect`.
2. O FastAPI valida o arquivo recebido.
3. A imagem é convertida para RGB e encaminhada ao YOLO11n.
4. O modelo identifica objetos das 10 classes selecionadas.
5. A API retorna as detecções em formato JSON.

### Configuração de inferência

Os parâmetros utilizados no modelo são:

- Modelo: YOLO11n;
- Confiança mínima: 20%;
- Resolução de inferência: 960 pixels;
- Dispositivo de execução: CPU;
- Classes: 10 classes selecionadas do COCO.

Esses parâmetros foram definidos após os testes de validação realizados durante o desenvolvimento.

## 6. Estrutura do projeto

```text
ClassDetect-AI/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── services/
│       ├── __init__.py
│       └── detector.py
│
├── datasets/
│   └── coco2017/
│
├── scripts/
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Descrição das pastas

**app/**

Contém a aplicação FastAPI e os componentes responsáveis pela detecção de objetos.

**app/main.py**

Define a aplicação FastAPI e seus endpoints.

**app/services/detector.py**

Carrega o modelo YOLO11n, executa as inferências e organiza os resultados das detecções.

**datasets/**

Organiza os dados do COCO 2017 utilizados nos experimentos. As imagens do dataset são mantidas localmente e não são incluídas no repositório Git.

**scripts/**

Contém scripts auxiliares utilizados para preparação do dataset, inferência e avaliação do modelo.

**tests/**

Contém os testes automatizados de integração da API.

**requirements.txt**

Registra as dependências Python necessárias para executar o projeto.

## 7. Organização do desenvolvimento

O desenvolvimento foi organizado com metodologia Kanban utilizando o Trello.

O versionamento utiliza Git e GitHub, com branches de desenvolvimento e Pull Requests para integração do código.

As principais branches são:

- `main`: versão principal do projeto.
- `develop`: integração das funcionalidades desenvolvidas.
- `feature/*`: desenvolvimento de funcionalidades e melhorias específicas.
```

## 8. Instalação do projeto

### Pré-requisitos

Para executar o ClassDetect AI, é necessário possuir:

- Python 3.12 instalado;
- Git instalado;
- Conexão com a internet para instalar as dependências;
- Um editor de código, como o Visual Studio Code.

### 8.1. Clonar o repositório

Abra um terminal e execute:

```bash
git clone https://github.com/miguelmagro-tech/ClassDetect-AI.git
cd ClassDetect-AI
```

### 8.2. Criar um ambiente virtual

No Windows, execute:

```powershell
py -3.12 -m venv .venv
```

Ative o ambiente virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

No Linux ou macOS, os comandos equivalentes são:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 8.3. Instalar as dependências

Com o ambiente virtual ativado, execute:

```bash
python -m pip install -r requirements.txt
```

Esse comando instalará as bibliotecas necessárias, incluindo FastAPI, Ultralytics, PyTorch, Pillow e Pytest.

**Observação:** na primeira execução do detector, o Ultralytics poderá baixar automaticamente os pesos pré-treinados `yolo11n.pt`, caso ainda não estejam disponíveis localmente.

## 9. Executando a API

Com o terminal aberto na raiz do projeto e o ambiente virtual ativado, execute:

```bash
python -m uvicorn app.main:app --reload
```

Após iniciar o servidor, acesse:

- Página inicial: http://127.0.0.1:8000/
- Documentação interativa: http://127.0.0.1:8000/docs

### 9.1. Testando pelo Swagger

O FastAPI disponibiliza uma interface interativa para testar os endpoints.

Para realizar uma detecção:

1. Acesse `http://127.0.0.1:8000/docs`.
2. Localize o endpoint `POST /detect`.
3. Clique em **Try it out**.
4. Selecione uma imagem no campo `file`.
5. Clique em **Execute**.
6. Observe o código HTTP e o JSON retornado.

### 9.2. Endpoints disponíveis

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Verifica o funcionamento da API |
| POST | `/detect` | Recebe uma imagem e retorna as detecções |

A API recebe imagens por meio de upload multipart/form-data.

O endpoint `/detect` retorna os objetos identificados, suas classes, os níveis de confiança e as coordenadas das caixas delimitadoras.

Arquivos inválidos ou imagens corrompidas retornam HTTP 400.

Requisições sem o arquivo obrigatório retornam HTTP 422.

### 9.3. Encerrando o servidor

Para interromper a execução da API no terminal, pressione:

`Ctrl + C`

## 10. Observações sobre o dataset

As imagens utilizadas na preparação e avaliação do dataset COCO 2017 não são incluídas no repositório GitHub devido ao volume dos arquivos.

Os dados são mantidos localmente na pasta `datasets/coco2017/`.

O funcionamento básico do endpoint `/detect` não exige que o usuário tenha o dataset COCO instalado, pois o sistema utiliza o modelo YOLO11n pré-treinado para processar imagens enviadas à API.


## 11. Exemplo de resposta da API

Ao enviar uma imagem válida para o endpoint `POST /detect`, a API retorna um objeto JSON com uma lista de detecções.

Exemplo de resposta:

```json
{
  "detections": [
    {
      "class_name": "laptop",
      "confidence": 0.4053,
      "bbox": {
        "x_min": 714,
        "y_min": 1,
        "x_max": 1083,
        "y_max": 230
      }
    },
    {
      "class_name": "book",
      "confidence": 0.3567,
      "bbox": {
        "x_min": 198,
        "y_min": 334,
        "x_max": 860,
        "y_max": 816
      }
    }
  ]
}
```

### Significado dos campos

| Campo | Descrição |
|-------|-----------|
| detections | Lista de objetos detectados |
| class_name | Nome da classe identificada |
| confidence | Confiança da previsão, entre 0 e 1 |
| bbox | Coordenadas da caixa delimitadora |
| x_min | Posição horizontal inicial |
| y_min | Posição vertical inicial |
| x_max | Posição horizontal final |
| y_max | Posição vertical final |

As coordenadas são expressas em pixels, considerando a imagem original.

Quando nenhum objeto das classes selecionadas é identificado, a resposta apresenta uma lista vazia:

```json
{
  "detections": []
}
```

## 12. Testes automatizados

O projeto utiliza Pytest e o TestClient do FastAPI para verificar o comportamento da API.

Os testes estão implementados no arquivo:

`tests/test_api.py`

### Cenários avaliados

| Teste | Verificação | HTTP esperado |
|-------|-------------|---------------|
| Imagem válida | Upload e resposta simulada do detector | 200 |
| Arquivo inválido | Rejeição de arquivo de texto | 400 |
| Arquivo ausente | Validação de campo obrigatório | 422 |
| Imagem corrompida | Tratamento de imagem inválida | 400 |
| YOLO11n real | Integração da API com o modelo | 200 |

O primeiro teste utiliza um mock para simular a resposta do detector, permitindo validar o endpoint de forma isolada.

O último teste utiliza o modelo YOLO11n real com uma imagem do subconjunto COCO disponível localmente.

### Executando os testes

Com o ambiente virtual ativado, execute na raiz do projeto:

```bash
python -m pytest tests/test_api.py -v
```

Durante o desenvolvimento, os cinco testes foram executados com sucesso.

**Observação:** o teste com YOLO11n real utiliza a imagem `000000001675.jpg`, armazenada localmente no dataset. Caso essa imagem não esteja disponível, o Pytest marca esse teste como `SKIPPED` (ignorado).

## 13. Validação do modelo

O modelo YOLO11n foi avaliado utilizando imagens selecionadas do dataset COCO 2017.

A avaliação utilizou as seguintes configurações:

- Modelo: YOLO11n pré-treinado;
- Imagens avaliadas: 173;
- Classes: 10 classes selecionadas;
- Confiança mínima: 0,20 (20%);
- Resolução de inferência: 960 pixels;
- IoU mínimo para correspondência: 0,50;
- Execução: CPU.

### Resultados obtidos

| Métrica | Resultado |
|---------|-----------|
| Precisão (Precision) | Aproximadamente 65,36% |
| Revocação (Recall) | Aproximadamente 57,62% |
| Verdadeiros positivos (TP) | 783 |
| Falsos positivos (FP) | 415 |
| Falsos negativos (FN) | 576 |

### Interpretação

A precisão representa a proporção das detecções realizadas pelo modelo que corresponderam corretamente às anotações de referência.

A revocação representa a proporção dos objetos anotados que foram identificados pelo modelo.

Os resultados demonstram a capacidade do YOLO11n de detectar objetos relacionados a ambientes de estudo, embora ainda existam detecções incorretas e objetos não identificados.

A avaliação foi realizada por meio de um procedimento simplificado de correspondência entre detecções e anotações, considerando classe e IoU mínimo de 0,50, sem considerar anotações do tipo `iscrowd`.

Portanto, esses resultados representam as métricas calculadas pelo projeto e não devem ser confundidos com o mAP oficial do benchmark COCO.

## 14. Limitações e possibilidades de melhoria

Apesar dos resultados alcançados, o sistema apresenta algumas limitações:

- A detecção está restrita às 10 classes selecionadas.
- A precisão pode variar de acordo com a iluminação, resolução e qualidade da imagem.
- Objetos pequenos, parcialmente ocultos ou sobrepostos podem não ser identificados.
- A inferência é executada em CPU, podendo ser mais lenta que em equipamentos com GPU.
- O modelo utiliza pesos pré-treinados, sem ajuste fino específico para ambientes de estudo.

Como melhorias futuras, podem ser considerados treinamento adicional, ampliação do dataset, otimização de desempenho e desenvolvimento de uma interface visual para utilização da API.
