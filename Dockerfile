# Use an official Python runtime
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy files from the App folder (not root)
COPY App/ .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask's default port
EXPOSE 5000

# Run your Flask app (change to main.py if needed)
CMD ["python", "web_app.py"]


