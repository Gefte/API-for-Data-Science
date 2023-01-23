
# Usando uma imagem do python como base
FROM python:3.9

# Definindo o diretório de trabalho
WORKDIR /app

# Copiando o código da API e o modelo para o container
COPY backend/main.py backend/
COPY backend/model.h5 backend/
COPY requirements.txt .

# Instalando as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expondo a porta 8000
EXPOSE 8000

# Definindo o comando para iniciar o servidor
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
