FROM python:3.9

WORKDIR /app

# Copy everything inside App folder to container's /app
COPY App/ /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "web_app.py"]  # or change to "main.py" if needed



