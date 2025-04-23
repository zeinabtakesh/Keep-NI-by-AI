FROM python:3.9

WORKDIR /app

COPY App/ /app/

RUN echo "✔ Here are the files inside /app:" && ls -la /app





