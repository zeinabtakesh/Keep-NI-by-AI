FROM python:3.9

WORKDIR /app

COPY . .

# Create a writable temp dir for pip
RUN mkdir -p /app/tmp

# Set temp env vars for pip
ENV TMPDIR=/app/tmp
ENV PIP_NO_CACHE_DIR=1

# Install dependencies using custom temp dir
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

EXPOSE 5000

CMD ["python", "web_app.py"]





