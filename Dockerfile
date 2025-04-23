FROM python:3.9

WORKDIR /app

# Copy everything from App folder into container's /app directory
COPY App/ /app/

# Debug: Print files in /app to verify requirements.txt was copied
RUN echo "Contents of /app:" && ls -l /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "web_app.py"]  # or main.py




