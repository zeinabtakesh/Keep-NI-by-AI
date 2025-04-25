FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy your app code first
COPY . .

# Install dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Install git to enable cloning
RUN apt-get update && apt-get install -y git

#  Clone the model from GitHub directly into the image
RUN git clone https://github.com/YOUR_USERNAME/Vit-GPT2-Offline.git /app/NourFakih-Vit-GPT2-UCA-UCF-07
ENV MODEL_PATH=/app/NourFakih-Vit-GPT2-UCA-UCF-07

# Expose app port
EXPOSE 5000

# Run your app
CMD ["python", "web_app.py"]
