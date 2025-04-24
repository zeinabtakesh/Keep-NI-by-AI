FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org -r requirements.txt
EXPOSE 5000
CMD ["python", "web_app.py"]





