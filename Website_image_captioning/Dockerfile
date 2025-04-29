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
# Create the model folder
# RUN mkdir -p /app/NourFakih-Vit-GPT2-UCA-UCF-07
# Move files after making sure destination folder exists
# RUN mkdir -p /app/NourFakih-Vit-GPT2-UCA-UCF-07 && \
#     mv "/app/repo-temp/The Model/"* /app/NourFakih-Vit-GPT2-UCA-UCF-07/
# Move ONLY the files from "The Model" into the target directory
# RUN mv "/app/repo-temp/Website_image_captioning/The Model"/* /app/NourFakih-Vit-GPT2-UCA-UCF-07/

## RUN mv "/app/repo-temp/The Model" /app/NourFakih-Vit-GPT2-UCA-UCF-07

# Set environment variable for the model path
#ENV MODEL_PATH=/app/NourFakih-Vit-GPT2-UCA-UCF-07

# Expose app port
EXPOSE 5000

# Run your app
CMD ["python", "web_app.py"]

# # Expose app port
# EXPOSE 5000

# # Run your app
# CMD ["python", "web_app.py"]
