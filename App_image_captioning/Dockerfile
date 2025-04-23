FROM python:3.9

WORKDIR /app

COPY . .

RUN mkdir -p /app/tmp
ENV TMPDIR=/app/tmp
ENV PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

EXPOSE 5000

CMD ["python", "web_app.py"]




