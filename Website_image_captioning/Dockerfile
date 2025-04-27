FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/zeinabtakesh/Keep-NI-by-AI.git /app/repo-temp

RUN mkdir -p /app/NourFakih-Vit-GPT2-UCA-UCF-07 && \
    cp -r "/app/repo-temp/The Model/"* /app/NourFakih-Vit-GPT2-UCA-UCF-07/

ENV MODEL_PATH=/app/NourFakih-Vit-GPT2-UCA-UCF-07

EXPOSE 5000

CMD ["python", "web_app.py"]
