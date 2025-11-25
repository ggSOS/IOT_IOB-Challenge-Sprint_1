# Sistema de Identificação Facial para Controle de Acesso
Este projeto implementa um sistema de detecção, reconhecimento e registro de rostos utilizando Dlib + OpenCV, permitindo abrir e fechar portas virtualmente com base na identificação facial. O sistema utiliza um banco de vetores faciais e realiza o reconhecimento em tempo real a partir de um vídeo ou câmera.


## Objetivos do Projeto
- Criar um sistema funcional de controle de acesso por reconhecimento facial
- Permitir o cadastro de novos usuários diretamente pela câmera
- Realizar detecção, extração de características (embedding) e identificação facial
- Controlar a “porta” (simulada no console), com abertura apenas para pessoas cadastradas
- Fornecer feedback visual e textual em tela para facilitar o uso


## Notas Éticas sobre o Uso de Dados Faciais
O presente trabalho utilizou imagens extraídas de um vídeo disponibilizado sem restrições de copyright. Apesar de não haver limitações legais explícitas quanto ao uso do material, é importante destacar algumas considerações éticas relacionadas ao uso de dados faciais:
### Minimização de Dados
O uso das imagens foi restrito ao mínimo necessário para fins acadêmicos de detecção e registro de rostos. Nenhuma tentativa foi feita de identificar as pessoas presentes no vídeo ou correlacionar suas imagens com dados pessoais ou comportamentais.
### Finalidade Exclusivamente Acadêmica
As imagens foram utilizadas exclusivamente para fins educacionais e de pesquisa, com o objetivo de estudar técnicas de detecção e alinhamento de faces.
Não houve redistribuição, comercialização ou uso do material para treinamentos de larga escala.
### Não Reidentificação
Durante o desenvolvimento do projeto, buscou-se evitar técnicas ou práticas que pudessem permitir a reidentificação dos indivíduos. O foco permaneceu apenas na análise técnica da detecção facial, sem coletar metadados ou realizar comparações externas.
### Considerações sobre Risco e Responsabilidade
O trabalho se compromete a manter as imagens e quaisquer dados derivados de forma segura e a não utilizá-los para fins que possam causar danos aos indivíduos retratados.
### Conformidade com Princípios Éticos
A condução do estudo seguiu princípios éticos amplamente aceitos em pesquisa computacional, incluindo:
- Respeito à privacidade
- Transparência no uso de dados
- Limitação de finalidade
- Não exploração de dados sensíveis


## Funcionamento Interno
### Limiar de reconhecimento
- THRESH = 0.5
    - Menor = mais preciso (menos falsos positivos)
    - Maior = mais permissivo (pode reconhecer errado)
### Detecção facial
- rects = detector(rgb, 0)
### Alinhamento facial
- chip = dlib.get_face_chip(rgb, shape)
### Extração do vetor de características (embedding)
- vec = rec.compute_face_descriptor(chip)
### Banco de Dados (db.pkl)
- Armazena:
{
"nome_pessoa": vetor_dlib_128d,
...
}

O arquivo é atualizado automaticamente ao cadastrar novos rostos.
### Comparação com o banco
- Calcula distância Euclidiana
- Se a distância for menor que THRESH, o rosto é reconhecido
### Lógica de Abertura/Fechamento de Porta
- A porta abre apenas se o modo escanear estiver ativo e reconhecer  um rosto registrado, ficando aberto por x segundos(definido pela variável door_cooldown)


## Arquitetura Geral
<pre>
├── IA_models/
│   ├── shape_predictor_5_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
│
├── database/
│   └── db.pkl   (gerado automaticamente)
│
├── assets/
│   ├── images/
│   │   └── print.png
│   └── videos/
│       ├── medium_ver_no_copyright_faces.MOV
│       └── small_ver_no_copyright_faces.MOV
│
└── main.py
</pre>


## Dependências
- Python 3.8+
- OpenCV
- Dlib
- NumPy
- Pickle, os, time (nativos do Python)
- Modelos .dat do Dlib

## Instalação das dependências
- python -m pip install dlib-bin opencv-python numpy

ou

- pip install -r .\requirements.txt

## Caso opte por utilizar câmera
- Liberar câmera:
    - net start audiosrv
- Alterar variável de vídeo para utilizar câmera:
    - cap = cv2.VideoCapture(0)


## Como Executar
- Utilize o vídeo existente ou coloque um vídeo em videos/ e altere o caminho em:
    - video_path = "videos/small_ver_no_copyright_faces.MOV"

- Execute:
    - python main.py

- Aguarde a inicialização do detector e do modelo.

- Use os comandos durante a execução:
    - 1 - Cadastrar novo rosto detectado
    - 2 - Ativar/desativar modo de escaneamento
    - 3 - Sair
