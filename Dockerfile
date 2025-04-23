# Use an official Python runtime
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy all files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask's default port
EXPOSE 5000

# Run your Flask app
CMD ["python", "web_app.py"]

