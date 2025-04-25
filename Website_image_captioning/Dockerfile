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

# Clone the full repo
RUN git clone https://github.com/zeinabtakesh/Keep-NI-by-AI.git /app/repo-temp

# Copy just the model folder to the correct location
# Move model from repo subfolder
RUN mv "/app/repo-temp/The Model" /app/NourFakih-Vit-GPT2-UCA-UCF-07

# Set environment variable for the model path
ENV MODEL_PATH=/app/NourFakih-Vit-GPT2-UCA-UCF-07

ENV OPENAI_API_KEY=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
# Expose app port
EXPOSE 5000

# Run your app
CMD ["python", "web_app.py"]
