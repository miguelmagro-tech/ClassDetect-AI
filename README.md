
# ClassDetect AI

Projeto Final — SC-TEC | Módulo 2

## Sobre o projeto

O ClassDetect AI é um sistema de Inteligência Artificial desenvolvido para detectar objetos em imagens de ambientes de estudo.

A aplicação utilizará YOLO11n para detecção de objetos e FastAPI para disponibilizar os resultados por meio de uma API REST.

## Dataset

Dataset selecionado: COCO 2017.

Classes utilizadas:
1. Pessoa
2. Mochila
3. Livro
4. Celular
5. Notebook
6. Cadeira
7. Garrafa
8. Copo
9. Teclado
10. Mouse

## Tecnologias

- Python
- YOLO11n
- Ultralytics
- FastAPI
- Git e GitHub
- Trello

## Funcionamento previsto

1. Receber uma imagem por meio de uma requisição POST.
2. Processar a imagem utilizando YOLO11n.
3. Identificar os objetos das 10 classes selecionadas.
4. Retornar as detecções em formato JSON, incluindo classe, confiança e coordenadas.

## Organização do projeto

O desenvolvimento utiliza a metodologia Kanban no Trello e versionamento com GitFlow.

## Status

Projeto em desenvolvimento.
  
