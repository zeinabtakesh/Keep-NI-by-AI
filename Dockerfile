# Use Python 3.9
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Flask default is 5000)
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]  # Replace "app.py" with your main file





